"""Independent-cohort follow-up of the nominated back-mixing precision block.

All checkpoints and calibration ranges remain frozen. The primary baseline is
the FFT deployment operator. Factorial float restorations locate sensitivity;
finite-bit probes are inference candidates, not retrained or synthesized IP.
"""
import argparse
import copy
import hashlib
import itertools
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch

from config import OfdmConfig
from eval_paired_b1_longcp_pdpfix import noise_residual, sha256_tensor
from ofdm import ChannelTorch, OfdmTxTorch
from paired_research_stats import summarize_paired_slots, summarize_seed_deltas
from quant import ActQ, QConfig, quant_complex_weight
from run_precision_localization_pdpfix import DiagnosticReceiver, file_hash, save_json
from struct_quant_model import StructuredQDCCN
from train import apply_fading_awgn

HERE = Path(__file__).resolve().parent
DEFAULT_OUT = HERE / "experiments/back_precision_followup_20261002"
BOUNDARIES = {
    "eq_back.q_in": "eq.m_back.q_in",
    "eq_back.q_out": "eq.m_back.q_out",
    "eq.q_div": "eq.q_div",
    "eq.q_ifft": "eq.q_ifft",
}
SPECS = {"baseline_fft": {}, "baseline_matrix": {"backend": "matrix"}}
for weight, qin, qout in itertools.product((False, True), repeat=3):
    if not any((weight, qin, qout)):
        continue
    key = "float_" + "_".join(k for k, v in (("w", weight), ("in", qin), ("out", qout)) if v)
    SPECS[key] = {"float_weight": weight,
                  "float_boundaries": [k for k, v in (("eq_back.q_in", qin), ("eq_back.q_out", qout)) if v]}
SPECS.update({
    "back_out_a12": {"bits": {"eq_back.q_out": 12}},
    "back_out_a16": {"bits": {"eq_back.q_out": 16}},
    "back_in_a16": {"bits": {"eq_back.q_in": 16}},
    "back_in_out_a16": {"bits": {"eq_back.q_in": 16, "eq_back.q_out": 16}},
    "div_back_in_a16": {"bits": {"eq.q_div": 16, "eq_back.q_in": 16}},
    "div_back_in_out_a16": {"bits": {"eq.q_div": 16, "eq_back.q_in": 16, "eq_back.q_out": 16}},
    "back_out_ifft_a16": {"bits": {"eq_back.q_out": 16, "eq.q_ifft": 16}},
})


class BackPrecisionReceiver(DiagnosticReceiver):
    def configure(self, variant):
        if variant not in SPECS:
            raise ValueError(variant)
        self.set_variant("w8a8_fft")
        self.spec = SPECS[variant]
        self.backend = self.spec.get("backend", "fft")
        modules = dict(self.named_modules())
        for name, bits in self.spec.get("bits", {}).items():
            module = modules[BOUNDARIES[name]]
            module.qr.bits = module.qi.bits = bits

    def aq(self, module, x, name, block=None):
        if name in self.spec.get("float_boundaries", ()):
            return x
        return super().aq(module, x, name, block)

    def mix(self, module, x, block):
        if block != "eq_back":
            return super().mix(module, x, block)
        x = self.aq(module.q_in, x, block + ".q_in", block)
        weight = module.weight if self.spec.get("float_weight", False) else quant_complex_weight(module.weight, 8, out_axis=1)
        return self.aq(module.q_out, x @ weight.to(x.dtype), block + ".q_out", block)


def point_seed(cohort, tag, seed, channel, snr):
    text = f"back-followup|{cohort}|{tag}|{seed}|{channel}|{snr:g}"
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:4], "little")


def run_point(model, original, cfg, tag, seed, channel_name, snr, args, output, old_hashes):
    started = time.perf_counter()
    rng_seed = point_seed(args.cohort, tag, seed, channel_name, snr)
    torch.manual_seed(rng_seed)
    channel = ChannelTorch(cfg, torch.device("cpu"), model=channel_name)
    assert channel.pdp_mode == "3gpp_db_to_linear_power"
    tx = OfdmTxTorch(cfg, torch.device("cpu"))
    with torch.inference_mode():
        transmitted, labels, _ = tx.transmit(args.slots)
        received, taps = apply_fading_awgn(channel, transmitted, torch.full((args.slots,), float(snr)))
        noise = noise_residual(transmitted, received, taps, channel)
        hashes = {name: sha256_tensor(value) for name, value in (
            ("transmitted", transmitted), ("labels", labels), ("received", received),
            ("taps", taps), ("noise_residual", noise))}
        if hashes["received"] in old_hashes:
            raise AssertionError("Received cohort duplicates an earlier screening condition")
        predictions = {}
        agreement = {"fft": 0.0, "matrix": 0.0}
        for variant in SPECS:
            model.configure(variant)
            pp = []
            for start in range(0, args.slots, args.batch):
                yy = received[start:start + args.batch]
                result = model(yy)
                if not torch.isfinite(result).all():
                    raise AssertionError(f"Non-finite output: {variant}")
                if variant.startswith("baseline_"):
                    backend = variant.split("_")[1]
                    original.set_transform_backend(backend)
                    expected = original(yy)
                    difference = float((result - expected).abs().max())
                    agreement[backend] = max(agreement[backend], difference)
                    if difference != 0:
                        raise AssertionError(f"Baseline differs from shared model: {backend}, {difference}")
                pp.append(result[..., 1] > .5)
            predictions[variant] = torch.cat(pp)
    baseline = predictions["baseline_fft"] != labels.bool()
    bits_per_slot = cfg.D_slot * cfg.m
    baseline_errors = baseline.reshape(args.slots, -1).sum(1).numpy()
    arrays = {"baseline_slot_errors": baseline_errors}
    summaries = {}
    for variant, predicted in predictions.items():
        errors = predicted != labels.bool()
        slot_errors = errors.reshape(args.slots, -1).sum(1).numpy()
        corrected = (baseline & ~errors).reshape(args.slots, -1).sum(1).numpy()
        harmed = (~baseline & errors).reshape(args.slots, -1).sum(1).numpy()
        flips = (predicted != predictions["baseline_fft"]).reshape(args.slots, -1).sum(1).numpy()
        assert np.array_equal(flips, corrected + harmed)
        assert np.array_equal(slot_errors - baseline_errors, harmed - corrected)
        arrays[variant + "__slot_errors"] = slot_errors
        arrays[variant + "__corrected"] = corrected
        arrays[variant + "__harmed"] = harmed
        arrays[variant + "__flips"] = flips
        summaries[variant] = summarize_paired_slots(
            slot_errors, baseline_errors, bits_per_slot,
            bootstrap_reps=0 if variant == "baseline_fft" else args.bootstrap,
            bootstrap_seed=rng_seed + 9901)
        summaries[variant].update(changed_decisions=int(flips.sum()),
                                  corrected_decisions=int(corrected.sum()),
                                  harmed_decisions=int(harmed.sum()))
    assert sha256_tensor(received) == hashes["received"]
    name = f"{tag}_seed{seed}_{channel_name}_{snr:g}dB"
    npz = output / (name + ".npz")
    np.savez_compressed(npz, **arrays)
    row = {"tag": tag, "training_seed": seed, "channel": channel_name, "snr_db": snr,
           "slots": args.slots, "bits_per_slot": bits_per_slot, "point_seed": rng_seed,
           "stimulus_sha256": hashes, "pdp_mode": channel.pdp_mode,
           "baseline_max_absolute_difference_all_slots": agreement,
           "variants": summaries, "slot_data": npz.name, "slot_data_sha256": file_hash(npz),
           "elapsed_seconds": time.perf_counter() - started}
    save_json(output / (name + ".json"), row)
    print(f"{name}: {row['elapsed_seconds']:.1f}s, FFT BER={baseline_errors.mean()/bits_per_slot:.6g}", flush=True)
    return row


def aggregate(rows, args):
    result = {"metric": "equal-weight mean BER difference over 2 channels x 4 SNRs per training seed",
              "reference": "baseline_fft", "uncertainty": "Three checkpoint/cohort seed contrasts; sample SD, not CI or hypothesis test",
              "tags": {}}
    for tag in args.tags:
        groups = {}
        for variant in SPECS:
            deltas = []
            for seed in args.seeds:
                selected = [r for r in rows if r["tag"] == tag and r["training_seed"] == seed]
                values = [r["variants"][variant]["candidate_minus_reference_ber"] for r in selected]
                deltas.append(float(np.mean(values)))
            groups[variant] = summarize_seed_deltas(deltas)
        result["tags"][tag] = groups
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT / "independent_2000")
    parser.add_argument("--tags", nargs="+", default=["b1_longcp", "b2_longcp", "b2_shortcp"])
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--channels", nargs="+", default=["EVA", "ETU"])
    parser.add_argument("--snrs", type=float, nargs="+", default=[12, 16, 20, 24])
    parser.add_argument("--cohort", type=int, default=2026100201)
    parser.add_argument("--slots", type=int, default=2000)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if min(args.slots, args.batch, args.threads) < 1 or args.slots < 2 or args.bootstrap < 0:
        parser.error("Invalid simulation sizes")
    output = args.output.resolve()
    if output.exists():
        parser.error("Use a new output directory; previous evidence is not overwritten")
    output.mkdir(parents=True)
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    old = HERE / "experiments/precision_localization_20261001/screen_2000"
    old_rows = [json.loads(p.read_text(encoding="utf-8")) for p in old.glob("*_seed*dB.json")]
    old_hashes = {r["stimulus_sha256"]["received"] for r in old_rows}
    manifest = {"status": "running", "smoke_not_for_publication": args.smoke,
                "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                "python": platform.python_version(), "python_executable": sys.executable,
                "torch": torch.__version__, "numpy": np.__version__, "device": "cpu",
                "variants_fixed_before_run": SPECS, "old_screening_conditions_checked": len(old_rows),
                "selection": "Back block nominated by previous screen; all listed variants fixed before this independent cohort. Exploratory follow-up, not external preregistration.",
                "scope": "Frozen QAT checkpoints and maxima; primary FFT baseline; weights 8 bits except explicit float diagnostics. No retraining, recalibration, HLS, bitstream or board claim.",
                "statistics": "Slot-paired conditional percentile bootstrap per contrast; no multiplicity-adjusted significance. Equal-weight grid means and three-seed sample SD are descriptive.",
                "source_hashes": {name: file_hash(HERE / name) for name in (
                    "run_back_precision_followup.py", "run_precision_localization_pdpfix.py",
                    "struct_quant_model.py", "quant.py", "quant_model.py", "struct_model.py",
                    "config.py", "ofdm.py", "train.py", "paired_research_stats.py",
                    "eval_paired_b1_longcp_pdpfix.py")}, "checkpoints": {}, "conditions": []}
    manifest_path = output / "manifest.json"
    save_json(manifest_path, manifest)
    rows = []
    try:
        for tag in args.tags:
            cfg = OfdmConfig(nbits=int(tag[1]), long_cp=tag.endswith("longcp"))
            for seed in args.seeds:
                path = HERE / "ckpt" / f"sq_{tag}_s_r0_slim_w8a8_pdpfix_seed{seed}.pt"
                state = torch.load(path, map_location="cpu", weights_only=True)
                model = BackPrecisionReceiver(cfg, rank=0, eq_dense=True, slim=True, qc=QConfig()).eval()
                model.load_state_dict(state)
                original = StructuredQDCCN(cfg, rank=0, eq_dense=True, slim=True, qc=QConfig()).eval()
                original.load_state_dict(state)
                if any(not bool(m.calibrated) for m in model.modules() if isinstance(m, ActQ)):
                    raise AssertionError("Missing frozen activation calibration")
                key = f"{tag}_seed{seed}"
                manifest["checkpoints"][key] = {"path": str(path), "sha256": file_hash(path)}
                for channel, snr in itertools.product(args.channels, args.snrs):
                    row = run_point(model, original, cfg, tag, seed, channel, snr, args, output, old_hashes)
                    rows.append(row)
                    manifest["conditions"].append(f"{tag}_seed{seed}_{channel}_{snr:g}dB.json")
                    save_json(manifest_path, manifest)
                if any(not torch.equal(value, state[name]) for name, value in model.state_dict().items()):
                    raise AssertionError("Frozen model state changed during inference")
                if file_hash(path) != manifest["checkpoints"][key]["sha256"]:
                    raise AssertionError("Original checkpoint changed")
        for name, digest in manifest["source_hashes"].items():
            if file_hash(HERE / name) != digest:
                raise AssertionError("Experiment source changed during run: " + name)
        save_json(output / "grid_summary.json", aggregate(rows, args))
        manifest.update(status="complete", unique_slots=sum(r["slots"] for r in rows),
                        unique_bits=sum(r["slots"] * r["bits_per_slot"] for r in rows),
                        all_source_and_checkpoint_hashes_unchanged=True)
    except BaseException as exc:
        manifest.update(status="failed_or_interrupted", error=repr(exc))
        save_json(manifest_path, manifest)
        raise
    save_json(manifest_path, manifest)
    print(json.dumps({k: manifest[k] for k in ("status", "unique_slots", "unique_bits")}), flush=True)


if __name__ == "__main__":
    main()
