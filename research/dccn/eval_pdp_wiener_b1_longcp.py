"""PDP-informed, non-instantaneous-CSI Wiener reference for BPSK/long CP.

The profile and sinc-resampling matrix are assumed known, but the estimator
sees only the sixteen received pilots. It never receives the realized channel
taps. Each generated test stream is checked against the archived eight-method
slot-paired experiment before its BER is accepted as a formal comparison.
"""

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch

from benchmarks import ConventionalReceiver, ofdm_frontend
from config import OfdmConfig
from eval_paired_b1_longcp_pdpfix import noise_residual, sha256_tensor
from eval_struct_qat import snr_at_ber
from ofdm import ChannelTorch, OfdmTxTorch
from train import apply_fading_awgn


HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
SNRS = (0., 4., 8., 12., 16., 20., 24., 28.)
SEEDS = (1, 2, 3)
CHANNELS = ("EVA", "ETU")
SLOTS = 2000
TARGET_BER = 0.02
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with open(temporary, "w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    os.replace(temporary, path)


def profile_covariance(channel, nfft):
    """R_H = F A^T diag(p) A F^H, matching ChannelTorch.sample_h."""
    A = channel.A.detach().cpu().numpy().astype(np.float64)
    p = channel.pdp.detach().cpu().numpy().astype(np.float64)
    taps = A.T @ (p[:, None] * A)
    indices = np.arange(nfft, dtype=np.float64)
    time = np.arange(A.shape[1], dtype=np.float64)
    fourier = np.exp(-2j * np.pi * indices[:, None] * time[None, :] / nfft)
    covariance = fourier @ taps @ fourier.conj().T
    return (covariance + covariance.conj().T) / 2


def wiener_matrix(channel, cfg, snr_db):
    """Map pilot LS observations to the full static-slot channel response."""
    R = profile_covariance(channel, cfg.K)
    k = np.asarray([sc for _, sc in cfg.pilot_pos], dtype=int)
    noise_var = 10 ** (-float(snr_db) / 10) / abs(cfg.PILOT_VALUE) ** 2
    observed = R[np.ix_(k, k)] + noise_var * np.eye(len(k))
    # Numerical jitter only; no validation-tuned regularization.
    observed += 1e-10 * np.eye(len(k))
    # Row-oriented solve: W = R[:, k] @ observed^{-1}.  The transpose on
    # `observed` is essential for complex Hermitian covariance; solving
    # observed @ X = R[k, :] would conjugate the cross-covariance.
    W = np.linalg.solve(observed.T, R[:, k].T).T  # [64, 16]
    return torch.from_numpy(W.T.astype(np.complex64)).to(DEVICE)


def run_seed(seed, cfg, tx, receiver, channel_models):
    archived = json.loads((RESULTS / f"paired_b1_longcp_pdpfix_seed{seed}.json")
                          .read_text(encoding="utf-8"))
    if archived["status"] != "FORMAL_CORRECTED_PDP_SLOT_PAIRED":
        raise AssertionError("Archived comparison is not formal")
    output = {"training_seed": seed, "channels": {}}
    pilot_syms = torch.tensor([s for s, _ in cfg.pilot_pos], device=DEVICE)
    pilot_sc = torch.tensor([k for _, k in cfg.pilot_pos], device=DEVICE)
    for channel_name in CHANNELS:
        channel = channel_models[channel_name]
        archived_points = archived["seed_result"]["channels"][channel_name]["points"]
        points = []
        for index, snr_db in enumerate(SNRS):
            point_seed = 20260904 + seed + int(snr_db * 10)
            torch.manual_seed(point_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(point_seed)
            with torch.no_grad():
                x_cp, labels, labels_pm1 = tx.transmit(SLOTS)
                received, taps = apply_fading_awgn(
                    channel, x_cp,
                    torch.full((SLOTS,), snr_db, device=DEVICE))
                residual = noise_residual(x_cp, received, taps, channel)
                hashes = {
                    "transmitted_time_samples": sha256_tensor(x_cp),
                    "information_bits": sha256_tensor(labels),
                    "channel_taps": sha256_tensor(taps),
                    "effective_noise_residual": sha256_tensor(residual),
                    "received_time_samples": sha256_tensor(received),
                }
                if hashes != archived_points[index]["stimulus_sha256"]:
                    raise AssertionError(f"Stream differs: seed={seed} "
                                         f"channel={channel_name} snr={snr_db}")
                Y = ofdm_frontend(received, cfg)
                pilot_ls = Y[:, pilot_syms, pilot_sc] / cfg.PILOT_VALUE
                H_hat = pilot_ls @ wiener_matrix(channel, cfg, snr_db)
                H_hat = H_hat.unsqueeze(1).expand(-1, cfg.N_SYM, -1)
                predicted = receiver.detect(Y, H_hat)
                errors = int((predicted != labels_pm1).sum().item())
            nbits = int(labels_pm1.numel())
            points.append({"snr_db": snr_db, "point_seed": point_seed,
                           "errors": errors, "bits": nbits,
                           "ber": errors / nbits,
                           "stimulus_sha256": hashes})
            print(f"seed={seed} {channel_name} {snr_db:g} dB: "
                  f"{errors}/{nbits} BER={errors / nbits:.6g}", flush=True)
        threshold = snr_at_ber(SNRS, [p["ber"] for p in points], TARGET_BER)
        output["channels"][channel_name] = {
            "points": points, "snr_at_target_db": threshold,
            "resampling_matrix_sha256": sha256_tensor(channel.A),
        }
    return output


def main():
    cfg = OfdmConfig(nbits=1, long_cp=True)
    tx = OfdmTxTorch(cfg, DEVICE)
    receiver = ConventionalReceiver(cfg, DEVICE)
    channels = {name: ChannelTorch(cfg, DEVICE, model=name)
                for name in CHANNELS}
    if any(ch.pdp_mode != "3gpp_db_to_linear_power" for ch in channels.values()):
        raise AssertionError("Corrected PDP required")
    results = [run_seed(seed, cfg, tx, receiver, channels) for seed in SEEDS]
    summary = {"status": "FORMAL_PDP_INFORMED_SLOT_PAIRED",
               "method": "known-profile covariance Wiener/LMMSE from 16 pilots",
               "assumptions": ["known EVA/ETU PDP and sinc-resampling matrix",
                               "slot-static channel", "no instantaneous channel taps",
                               "diagonal AWGN pilot noise model; ignores finite-CP interference"],
               "protocol": {"seeds": SEEDS, "channels": CHANNELS, "snrs_db": SNRS,
                            "slots_per_snr_seed": SLOTS, "target_ber": TARGET_BER,
                            "source_paired_result":
                                "results/paired_b1_longcp_pdpfix_seed{1,2,3}.json",
                            "script_sha256": file_hash(__file__)},
               "per_seed": results,
               "summary": {}}
    for channel_name in CHANNELS:
        values = [r["channels"][channel_name]["snr_at_target_db"]
                  for r in results]
        finite = [v for v in values if v is not None]
        summary["summary"][channel_name] = {
            "snr_at_target_per_seed_db": values,
            "mean_db": float(np.mean(finite)) if len(finite) == len(SEEDS) else None,
            "sample_sd_db": float(np.std(finite, ddof=1))
                            if len(finite) == len(SEEDS) else None,
        }
    path = RESULTS / "pdp_wiener_b1_longcp_pdpfix_summary.json"
    if path.exists():
        raise FileExistsError(path)
    save_json(path, summary)
    print("saved", path)


if __name__ == "__main__":
    main()
