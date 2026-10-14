"""Strictly slot-paired corrected-PDP BPSK/long-CP BER evaluation.

For each channel, SNR, and seed, this evaluator calls the transmitter and
channel exactly once, then presents the same received samples to all three
learned receivers and all five conventional references. The stored hashes and
paired error tables make the common-random-number protocol auditable.

Formal use (requires all 12 corrected-PDP checkpoints):
    python eval_paired_b1_longcp_pdpfix.py

Small code-path check while retraining is in progress (legacy checkpoints are
used only to exercise the evaluator, never as a formal result):
    python eval_paired_b1_longcp_pdpfix.py --smoke --legacy-checkpoints
"""

import argparse
import hashlib
import json
import os
import platform
import sys
from pathlib import Path

import numpy as np
import torch

from benchmarks import ConventionalReceiver, ofdm_frontend
from config import OfdmConfig
from eval_struct_qat import build_model, snr_at_ber
from ofdm import ChannelTorch, OfdmTxTorch
from train import apply_fading_awgn


HERE = Path(__file__).resolve().parent
CKPT_DIR = HERE / "ckpt"
RESULTS_DIR = HERE / "results"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LEARNED = ("dense_fp32", "structured_fp32", "structured_w8a8")
CLASSICAL = ("ls_spline", "ideal_lmmse", "almmse", "ls_cp", "perfect_csi")
METHODS = LEARNED + CLASSICAL
PAIR_KEYS = (
    ("structured_fp32", "dense_fp32"),
    ("structured_w8a8", "structured_fp32"),
    ("structured_w8a8", "ls_spline"),
    ("structured_w8a8", "ideal_lmmse"),
    ("structured_w8a8", "almmse"),
    ("structured_w8a8", "ls_cp"),
    ("structured_w8a8", "perfect_csi"),
)


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_tensor(tensor):
    """Hash the tensor's exact stored values, dtype, and shape."""
    array = tensor.detach().contiguous().cpu().numpy()
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()


def checkpoint_paths(seed, legacy):
    suffix = "" if legacy else f"_pdpfix_seed{seed}"
    tag = "b1_longcp"
    return {
        "basic_fp32": CKPT_DIR / f"basic_{tag}{suffix}.pt",
        "dense_fp32": CKPT_DIR / f"dccn_{tag}{suffix}.pt",
        "structured_fp32": CKPT_DIR / f"struct_{tag}_s_r0_slim{suffix}.pt",
        "structured_w8a8": CKPT_DIR / f"sq_{tag}_s_r0_slim_w8a8{suffix}.pt",
    }


def preflight(seeds, legacy, output_dir, smoke, force):
    missing = [(seed, str(path)) for seed in seeds
               for path in checkpoint_paths(seed, legacy).values()
               if not path.is_file()]
    if missing:
        detail = "\n".join(f"  seed {seed}: {path}" for seed, path in missing)
        raise FileNotFoundError(f"Required checkpoints are not ready:\n{detail}")
    prefix = ("paired_b1_longcp_smoke_legacy" if legacy
              else "paired_b1_longcp_smoke_corrected" if smoke
              else "paired_b1_longcp_pdpfix")
    outputs = [output_dir / f"{prefix}_seed{seed}.json" for seed in seeds]
    outputs.append(output_dir / f"{prefix}_summary.json")
    existing = [str(path) for path in outputs if path.exists()]
    if existing and not force:
        raise FileExistsError("Output exists; pass --force to replace:\n  "
                              + "\n  ".join(existing))
    return prefix


def load_learned(cfg, seed, legacy, tx, channels):
    suffix = "" if legacy else f"_pdpfix_seed{seed}"
    return {
        "dense_fp32": build_model("float", cfg, "b1_longcp", tx, channels,
                                  checkpoint_suffix=suffix,
                                  float_checkpoint_suffix=suffix),
        "structured_fp32": build_model("s_r0_slim", cfg, "b1_longcp", tx,
                                       channels, checkpoint_suffix=suffix),
        "structured_w8a8": build_model("qat_s_r0_slim_w8a8", cfg,
                                       "b1_longcp", tx, channels,
                                       checkpoint_suffix=suffix),
    }


def noise_residual(x_cp, y_cp, taps, channel):
    """Recover the effective stored AWGN residual without another RNG call."""
    n = x_cp.shape[0]
    x = x_cp.reshape(n, -1)
    if channel.L == 1:
        clean = x * taps
    else:
        nfft = 1 << (x.shape[1] + channel.L - 2).bit_length()
        clean = torch.fft.ifft(torch.fft.fft(x, n=nfft, dim=1)
                               * torch.fft.fft(taps, n=nfft, dim=1),
                               dim=1)[:, :x.shape[1]]
    return y_cp.reshape(n, -1) - clean


def learned_error_masks(models, received, labels, model_batch):
    masks = {}
    count = labels.shape[0]
    for name, model in models.items():
        mask = torch.empty(labels.shape, dtype=torch.bool, device=DEVICE)
        with torch.no_grad():
            for start in range(0, count, model_batch):
                stop = min(start + model_batch, count)
                logits = model(received[start:stop])
                prediction = (logits[..., 1] > 0.5).to(labels.dtype)
                if prediction.shape != labels[start:stop].shape:
                    raise ValueError((name, prediction.shape,
                                      labels[start:stop].shape))
                mask[start:stop] = prediction != labels[start:stop]
        masks[name] = mask
    return masks


def classical_error_masks(cfg, receiver, received, taps, labels_pm1,
                          snr_db, cp_batch):
    with torch.no_grad():
        Y = ofdm_frontend(received, cfg)
        H_ls = receiver.ls_spline(Y)
        H_true = torch.fft.fft(taps, n=cfg.K, dim=1)
        H_true = H_true.unsqueeze(1).expand(-1, cfg.N_SYM, -1)
        guesses = {
            "ls_spline": receiver.detect(Y, H_ls),
            "ideal_lmmse": receiver.detect(
                Y, receiver.ideal_lmmse(H_ls, H_true, snr_db)),
            "almmse": receiver.detect(
                Y, receiver.approx_lmmse(H_ls, snr_db)),
            "perfect_csi": receiver.detect(Y, H_true.contiguous()),
        }
        cp_guesses = []
        for start in range(0, received.shape[0], cp_batch):
            stop = min(start + cp_batch, received.shape[0])
            equalized = receiver.ls_cp_equalize(
                Y[start:stop], H_ls[start:stop], received[start:stop])
            cp_guesses.append(receiver.detect_equalized(equalized))
        guesses["ls_cp"] = torch.cat(cp_guesses, dim=0)
        masks = {}
        for name in CLASSICAL:
            if guesses[name].shape != labels_pm1.shape:
                raise ValueError((name, guesses[name].shape, labels_pm1.shape))
            masks[name] = guesses[name] != labels_pm1
    return masks


def paired_counts(a, b, nbits):
    a_only = int((a & ~b).sum().item())
    b_only = int((~a & b).sum().item())
    both = int((a & b).sum().item())
    neither = nbits - a_only - b_only - both
    if min(a_only, b_only, both, neither) < 0:
        raise AssertionError("Invalid paired contingency counts")
    return {"a_only_error": a_only, "b_only_error": b_only,
            "both_error": both, "neither_error": neither}


def run_condition(cfg, tx, channel, receiver, models, snr_db, seed, slots,
                  model_batch, cp_batch):
    eval_seed = 20260904 + seed
    point_seed = eval_seed + int(snr_db * 10)
    torch.manual_seed(point_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(point_seed)
    with torch.no_grad():
        # One generation call is the central pairing invariant: no method is
        # permitted to regenerate bits, channel taps, or noise independently.
        x_cp, labels, labels_pm1 = tx.transmit(slots)
        snr_values = torch.full((slots,), float(snr_db), device=DEVICE)
        received, taps = apply_fading_awgn(channel, x_cp, snr_values)
        if channel.pdp_mode != "3gpp_db_to_linear_power":
            raise AssertionError(f"Unexpected PDP mode: {channel.pdp_mode}")
        residual = noise_residual(x_cp, received, taps, channel)
        stimulus_hashes = {
            "transmitted_time_samples": sha256_tensor(x_cp),
            "information_bits": sha256_tensor(labels),
            "channel_taps": sha256_tensor(taps),
            "effective_noise_residual": sha256_tensor(residual),
            "received_time_samples": sha256_tensor(received),
        }
        masks = learned_error_masks(models, received, labels, model_batch)
        masks.update(classical_error_masks(
            cfg, receiver, received, taps, labels_pm1, snr_db, cp_batch))
        if (sha256_tensor(received) != stimulus_hashes["received_time_samples"]
                or sha256_tensor(labels) != stimulus_hashes["information_bits"]
                or sha256_tensor(taps) != stimulus_hashes["channel_taps"]):
            raise AssertionError("An evaluator mutated the shared stimulus")
        nbits = int(labels.numel())
        if set(masks) != set(METHODS):
            raise AssertionError((set(masks), set(METHODS)))
        errors = {name: int(masks[name].sum().item()) for name in METHODS}
        paired = {f"{a}_vs_{b}": paired_counts(masks[a], masks[b], nbits)
                  for a, b in PAIR_KEYS}
        for a, b in PAIR_KEYS:
            item = paired[f"{a}_vs_{b}"]
            if item["a_only_error"] + item["both_error"] != errors[a]:
                raise AssertionError(f"Paired count mismatch: {a}")
            if item["b_only_error"] + item["both_error"] != errors[b]:
                raise AssertionError(f"Paired count mismatch: {b}")
    return {"snr_db": float(snr_db), "point_seed": point_seed,
            "slots": slots, "bits": nbits, "errors": errors,
            "ber": {name: errors[name] / nbits for name in METHODS},
            "stimulus_sha256": stimulus_hashes,
            "paired_error_counts": paired}


def save_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with open(temporary, "w", encoding="utf-8") as stream:
        json.dump(data, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
    os.replace(temporary, path)


def summarize(seed_results, channels, snrs, target_ber):
    summary = {"n_seeds": len(seed_results), "target_ber": target_ber,
               "channels": {}}
    for channel in channels:
        methods = {}
        for method in METHODS:
            ordered_seeds = sorted(map(int, seed_results))
            per_seed = [seed_results[str(seed)]["channels"][channel]
                        ["snr_at_target_db"][method]
                        for seed in ordered_seeds]
            finite = [value for value in per_seed if value is not None]
            per_seed_ber = [[point["ber"][method] for point in
                             seed_results[str(seed)]["channels"][channel]
                             ["points"]] for seed in ordered_seeds]
            per_seed_errors = [[point["errors"][method] for point in
                                seed_results[str(seed)]["channels"][channel]
                                ["points"]] for seed in ordered_seeds]
            ber_array = np.asarray(per_seed_ber, dtype=float)
            methods[method] = {
                "per_seed_snr_at_target_db": per_seed,
                "n_crossed": len(finite),
                "mean_db": float(np.mean(finite)) if len(finite) == len(per_seed)
                           else None,
                "sample_sd_db": float(np.std(finite, ddof=1))
                                if len(finite) == len(per_seed)
                                and len(finite) > 1 else None,
                "per_seed_errors": per_seed_errors,
                "per_seed_ber": per_seed_ber,
                "ber_mean": np.mean(ber_array, axis=0).tolist(),
                "ber_sample_sd": (np.std(ber_array, axis=0, ddof=1).tolist()
                                  if len(per_seed_ber) > 1 else None),
            }
        deltas = {}
        for a, b in PAIR_KEYS:
            per_seed = [None if methods[a]["per_seed_snr_at_target_db"][i] is None
                        or methods[b]["per_seed_snr_at_target_db"][i] is None
                        else methods[a]["per_seed_snr_at_target_db"][i]
                        - methods[b]["per_seed_snr_at_target_db"][i]
                        for i in range(len(seed_results))]
            finite = [value for value in per_seed if value is not None]
            deltas[f"{a}_minus_{b}"] = {
                "per_seed_db": per_seed,
                "mean_db": float(np.mean(finite)) if len(finite) == len(per_seed)
                           else None,
                "sample_sd_db": float(np.std(finite, ddof=1))
                                if len(finite) == len(per_seed)
                                and len(finite) > 1 else None,
            }
        summary["channels"][channel] = {
            "snrs_db": snrs, "methods": methods,
            "paired_threshold_deltas": deltas}
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true",
                        help="Run a small test that cannot be used for publication")
    parser.add_argument("--legacy-checkpoints", action="store_true",
                        help="Smoke-only: exercise the code with pre-PDP-fix weights")
    parser.add_argument("--seeds", default=None)
    parser.add_argument("--channels", default=None)
    parser.add_argument("--snrs", default=None)
    parser.add_argument("--slots", type=int, default=None)
    parser.add_argument("--model-batch", type=int, default=250)
    parser.add_argument("--cp-batch", type=int, default=100)
    parser.add_argument("--target-ber", type=float, default=0.02)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if args.legacy_checkpoints and not args.smoke:
        parser.error("--legacy-checkpoints is allowed only with --smoke")
    seeds = [int(value) for value in (args.seeds or ("1" if args.smoke
                                                    else "1,2,3")).split(",")]
    channels = (args.channels or ("ETU" if args.smoke else "EVA,ETU")).split(",")
    snrs = [float(value) for value in
            (args.snrs or ("12" if args.smoke else "0,4,8,12,16,20,24,28"))
            .split(",")]
    slots = args.slots if args.slots is not None else (4 if args.smoke else 2000)
    output_dir = args.output_dir or (RESULTS_DIR / "paired_smoke"
                                      if args.smoke else RESULTS_DIR)
    if not seeds or len(seeds) != len(set(seeds)) or min(seeds) < 1:
        parser.error("--seeds must be distinct positive integers")
    if any(channel not in ("EVA", "ETU") for channel in channels):
        parser.error("--channels must contain only EVA and/or ETU")
    if slots < 1 or args.model_batch < 1 or args.cp_batch < 1:
        parser.error("slots and batch sizes must be positive")
    if not 0 < args.target_ber < 1:
        parser.error("--target-ber must lie between zero and one")
    if not args.smoke and (seeds != [1, 2, 3]
                           or channels != ["EVA", "ETU"]
                           or snrs != [0., 4., 8., 12., 16., 20., 24., 28.]
                           or slots != 2000 or args.target_ber != 0.02):
        parser.error("Formal protocol is fixed at 3 seeds, EVA/ETU, "
                     "0:4:28 dB, 2000 slots/SNR, BER target 0.02")
    prefix = preflight(seeds, args.legacy_checkpoints, output_dir,
                       args.smoke, args.force)

    cfg = OfdmConfig(nbits=1, long_cp=True)
    tx = OfdmTxTorch(cfg, DEVICE)
    channel_models = {name: ChannelTorch(cfg, DEVICE, model=name)
                      for name in channels}
    if any(model.pdp_mode != "3gpp_db_to_linear_power"
           for model in channel_models.values()):
        raise AssertionError("The formal evaluator requires corrected PDPs")
    receiver = ConventionalReceiver(cfg, DEVICE)
    channel_profiles = {
        name: {
            "pdp_db": model.pdp_db.detach().cpu().tolist(),
            "pdp_linear_normalized": model.pdp.detach().cpu().tolist(),
            "resampling_matrix_sha256": sha256_tensor(model.A),
        } for name, model in channel_models.items()
    }
    seed_results = {}
    print(f"device={DEVICE}, slots/SNR={slots}, channels={channels}, "
          f"seeds={seeds}, status={'SMOKE' if args.smoke else 'FORMAL'}",
          flush=True)
    for seed in seeds:
        paths = checkpoint_paths(seed, args.legacy_checkpoints)
        checkpoint_hashes = {name: sha256_file(path)
                             for name, path in paths.items()}
        models = load_learned(cfg, seed, args.legacy_checkpoints,
                              tx, channel_models)
        channel_results = {}
        for channel_name in channels:
            channel = channel_models[channel_name]
            points = []
            for snr in snrs:
                point = run_condition(cfg, tx, channel, receiver, models, snr,
                                      seed, slots, args.model_batch,
                                      args.cp_batch)
                points.append(point)
                score = " ".join(f"{name}={point['ber'][name]:.4g}"
                                 for name in METHODS)
                print(f"seed={seed} {channel_name} {snr:g} dB {score}",
                      flush=True)
            threshold = {name: snr_at_ber(
                snrs, [point["ber"][name] for point in points],
                args.target_ber) for name in METHODS}
            channel_results[channel_name] = {
                "snrs_db": snrs,
                "points": points,
                "snr_at_target_db": threshold,
            }
        seed_record = {
            "training_seed": seed,
            "evaluation_seed": 20260904 + seed,
            "checkpoint_paths": {name: str(path) for name, path in paths.items()},
            "checkpoint_sha256": checkpoint_hashes,
            "channels": channel_results,
        }
        seed_results[str(seed)] = seed_record
        output = output_dir / f"{prefix}_seed{seed}.json"
        save_json(output, {
            "status": "SMOKE_NOT_FOR_PUBLICATION" if args.smoke
                      else "FORMAL_CORRECTED_PDP_SLOT_PAIRED",
            "protocol": {
                "system": cfg.summary(),
                "pdp_mode": "3gpp_db_to_linear_power",
                "channel_profiles": channel_profiles,
                "slots_per_snr_per_seed": slots,
                "bits_per_slot": cfg.D_slot * cfg.m,
                "snr_grid_db": snrs,
                "target_ber": args.target_ber,
                "evaluation_point_seed_rule":
                    "20260904 + training_seed + int(snr_db * 10), reset per channel/SNR",
                "generation": "one transmit and one fading/AWGN call per channel/SNR/seed; every method consumes the identical received tensor",
                "threshold_interpolation":
                    "per-seed nonincreasing BER envelope, first crossing, linear in SNR versus log10(BER)",
                "ideal_lmmse": "oracle true-channel direction",
                "legacy_checkpoints": args.legacy_checkpoints,
                "device": str(DEVICE),
                "torch_version": torch.__version__,
                "numpy_version": np.__version__,
                "python_version": platform.python_version(),
                "script_sha256": sha256_file(__file__),
            },
            "seed_result": seed_record,
        })
        print("saved", output, flush=True)
        del models
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    aggregate = summarize(seed_results, channels, snrs, args.target_ber)
    summary_output = output_dir / f"{prefix}_summary.json"
    save_json(summary_output, {
        "status": "SMOKE_NOT_FOR_PUBLICATION" if args.smoke
                  else "FORMAL_CORRECTED_PDP_SLOT_PAIRED",
        "seed_result_paths": [str(output_dir / f"{prefix}_seed{seed}.json")
                              for seed in seeds],
        "summary": aggregate,
    })
    print("saved", summary_output, flush=True)


if __name__ == "__main__":
    try:
        main()
    except (AssertionError, FileExistsError, FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
