"""Fixed-checkpoint, slot-paired precision localization (exploratory only).

Restorations retain learned QAT weights, biases, delta and frozen activation
amax buffers. They are inference counterfactuals, not retrained mixed-precision
receivers, nor RTL/board validation. Each condition generates traffic once.
"""

import argparse
import copy
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from config import OfdmConfig
from eval_paired_b1_longcp_pdpfix import noise_residual, sha256_tensor
from ofdm import ChannelTorch, OfdmTxTorch
from paired_research_stats import summarize_paired_slots
from quant import (ActQ, QConfig, cdiv_fixed, qmax_of, quant_complex_weight,
                   quant_real_weight, rms_norm_fixed)
from struct_model import StructuredDCCN
from struct_quant_model import (StructuredQDCCN, cp_expand_ifft, cp_fold_fft,
                               quant_complex_vector)
from train import apply_fading_awgn

HERE = Path(__file__).resolve().parent
DEFAULT_OUT = HERE / "experiments" / "precision_localization_20261001"
BLOCKS = ("eq_front", "eq_back", "chest_in", "chest_res0", "chest_res1",
          "chest_out", "basic_gain", "classifier")
VARIANTS = ("w8a8_matrix", "w8a8_fft", "restore_norm", "restore_div",
            "restore_tanh") + tuple("restore_" + b for b in BLOCKS) + (
                "restore_all_weights", "restore_all_activations", "w8a16",
                "w16a8", "restore_nonlinear_operators", "restore_mixing", "restore_chest",
                "qat_weights_all_float", "structured_fp32")
PRIMARY_CONTRASTS = {"w8a8_fft", "restore_norm", "restore_div", "restore_tanh",
                     "restore_nonlinear_operators", "restore_mixing", "restore_chest",
                     "restore_classifier", "qat_weights_all_float", "structured_fp32"}


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_json(path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def metrics(a, b, count_outside=None):
    """Per-slot sum-squares sufficient statistics; slots are the cluster unit."""
    batch = a.shape[0]
    error = (a - b).abs().square().reshape(batch, -1)
    signal = b.abs().square().reshape(batch, -1)
    outside = (torch.zeros(batch, dtype=torch.int64, device=a.device)
               if count_outside is None else
               count_outside.reshape(batch, -1).sum(dim=1))
    return torch.stack((error.sum(1), signal.sum(1), error.amax(1),
                        outside.to(error.dtype),
                        torch.full((batch,), error.shape[1], device=a.device)), 1)


class DiagnosticReceiver(StructuredQDCCN):
    """Operation-order-preserving baseline with explicit precision interventions."""

    def set_variant(self, variant):
        self.variant = variant
        self.restore_blocks = set()
        self.float_norm = self.float_div = self.float_tanh = False
        self.float_w = self.float_a = False
        self.trace_enabled = False
        self.local = {}
        self.stages = {}
        self.backend = "fft" if variant == "w8a8_fft" else "matrix"
        self.wbits = 16 if variant == "w16a8" else 8
        self.abits = 16 if variant == "w8a16" else 8
        if variant.startswith("restore_") and variant[8:] in BLOCKS:
            self.restore_blocks.add(variant[8:])
        self.float_norm = variant in ("restore_norm", "restore_nonlinear_operators",
                                      "qat_weights_all_float")
        self.float_div = variant in ("restore_div", "restore_nonlinear_operators",
                                     "qat_weights_all_float")
        self.float_tanh = variant in ("restore_tanh", "restore_nonlinear_operators",
                                      "qat_weights_all_float")
        self.float_w = variant in ("restore_all_weights", "qat_weights_all_float")
        self.float_a = variant in ("restore_all_activations", "qat_weights_all_float")
        if variant == "restore_chest":
            self.restore_blocks.update(("chest_in", "chest_res0", "chest_res1",
                                        "chest_out"))
        if variant == "restore_mixing":
            self.restore_blocks.update(("eq_front", "eq_back"))
        for module in self.modules():
            if isinstance(module, ActQ):
                module.bits = self.abits

    def mark(self, name, x):
        if self.trace_enabled:
            self.stages[name] = x.detach()
        return x

    def record(self, name, actual, exact, outside=None):
        if self.trace_enabled:
            self.local[name] = metrics(actual, exact, outside).detach()
        return actual

    def aq(self, module, x, name, block=None):
        if self.float_a or block in self.restore_blocks:
            return x
        out = module(x)
        if self.trace_enabled:
            if isinstance(module, ActQ):
                bound = float(module.amax) * 1.02
                self.record(name, out, x, x.abs() > bound)
            else:
                self.record(name + ".real", out.real, x.real,
                            x.real.abs() > float(module.qr.amax) * 1.02)
                self.record(name + ".imag", out.imag, x.imag,
                            x.imag.abs() > float(module.qi.amax) * 1.02)
        return out

    def linear(self, module, x, block):
        x = self.aq(module.q_in, x, block + ".q_in", block)
        weight = module.weight if (self.float_w or block in self.restore_blocks) \
            else quant_real_weight(module.weight, self.wbits)
        out = F.linear(x, weight, module.bias)
        return self.aq(module.q_out, out, block + ".q_out", block)

    def mix(self, module, x, block):
        if not module.dense:
            raise NotImplementedError("This audit requires dense equalizer mixing.")
        x = self.aq(module.q_in, x, block + ".q_in", block)
        weight = module.weight if (self.float_w or block in self.restore_blocks) \
            else quant_complex_weight(module.weight, self.wbits, out_axis=1)
        out = x @ weight.to(x.dtype)
        return self.aq(module.q_out, out, block + ".q_out", block)

    def norm(self, x, block):
        qc = copy.copy(self.qc)
        qc.rsqrt = "float" if self.float_norm else self.qc.rsqrt
        out = rms_norm_fixed(x, qc)
        exact = x * (x.abs().square().mean((1, 2), keepdim=True) + 1e-8).sqrt().reciprocal()
        return self.record(block + ".rms", out, exact)

    def forward(self, received):
        self.local = {}
        self.stages = {}
        cfg, eq, basic = self.cfg, self.eq, self.basic
        batch = received.shape[0]
        y = self.norm(received, "eq")
        y = self.aq(eq.q_norm, y, "eq.q_norm")
        yf = cp_fold_fft(y, cfg.CP) if self.backend == "fft" else \
            torch.einsum('bfs,sn->bfn', y, eq.Fd.to(y.dtype))
        yf = self.mark("eq.transform_prequant", yf)
        yf = self.aq(eq.q_fft, yf, "eq.q_fft")
        Y = self.mark("eq.front", self.mix(eq.m_front, yf, "eq_front")
                      + eq.b_front.to(y.dtype))
        q = torch.cat([Y.real, Y.imag], -1).reshape(batch, -1)
        h = self.linear(eq.c_in, q, "chest_in")
        hs, x = [], h
        for i, lin in enumerate(eq.c_res):
            pre = self.linear(lin, x, f"chest_res{i}")
            value = torch.tanh(pre) if self.float_tanh else eq.act(pre)
            self.record(f"chest_res{i}.tanh", value, torch.tanh(pre),
                        pre.abs() > eq.act.R)
            hs.append(value)
            x = (hs[-2] if len(hs) >= 2 else h) - hs[-1]
        hq = self.linear(eq.c_out, torch.cat(hs, -1), "chest_out")
        hq = hq.view(batch, cfg.N_SYM, cfg.K, 2)
        H = self.mark("eq.channel", torch.complex(hq[..., 0], hq[..., 1]))
        divqc = copy.copy(self.qc)
        divqc.div = "float" if self.float_div else self.qc.div
        divqc.abits = self.abits
        X = cdiv_fixed(Y, H, eq.delta.abs(), divqc)
        Xexact = Y * H.conj() * (H.abs().square() + eq.delta.abs()).clamp_min(1e-12).reciprocal()
        self.record("eq.div", X, Xexact)
        X = self.mark("eq.div_prequant", X)
        X = self.aq(eq.q_div, X, "eq.q_div")
        t = self.mark("eq.back", self.mix(eq.m_back, X, "eq_back"))
        x_hat = (cp_expand_ifft(t, cfg.CP) if self.backend == "fft" else
                 t @ eq.FdcT.to(t.dtype)) + eq.b_back.to(t.dtype)
        x_hat = self.mark("eq.inverse_prequant", x_hat)
        x_hat = self.aq(eq.q_ifft, x_hat, "eq.q_ifft")
        y = self.norm(x_hat, "basic")
        y = self.aq(basic.q_norm, y, "basic.q_norm")
        Yb = cp_fold_fft(y, cfg.CP) if self.backend == "fft" else \
            torch.einsum('bfs,sn->bfn', y, basic.Fd.to(y.dtype))
        Yb = self.mark("basic.transform_prequant", Yb)
        Yb = self.aq(basic.q_fft, Yb, "basic.q_fft")
        gain = basic.g if (self.float_w or "basic_gain" in self.restore_blocks) \
            else quant_complex_vector(basic.g, self.wbits)
        d = Yb[:, basic.f_idx, basic.k_idx] * gain.to(y.dtype) + basic.bias.to(y.dtype)
        if basic.rank != 0:
            raise NotImplementedError("This audit requires rank-zero basic receiver.")
        d = self.mark("basic.gain_prequant", d)
        d = self.aq(basic.q_out, d, "basic.q_out", "basic_gain")
        iq = torch.stack([d.real, d.imag], -1)
        feat = torch.cat([iq, F.leaky_relu(iq)], -1)
        logits = F.leaky_relu(self.linear(basic.classifier, feat, "classifier"))
        logits = self.mark("basic.logits", logits.view(batch, cfg.D_slot, cfg.m, 2))
        return torch.softmax(logits, -1)


def static_weight_metrics(model):
    """Quantization distortion of the fixed checkpoint, not a fitting metric."""
    result = {}
    weights = {"eq_front": (model.eq.m_front.weight, "complex", 1),
               "eq_back": (model.eq.m_back.weight, "complex", 1),
               "basic_gain": (model.basic.g, "vector", None)}
    for name, module in (("chest_in", model.eq.c_in),
                         ("chest_res0", model.eq.c_res[0]),
                         ("chest_res1", model.eq.c_res[1]),
                         ("chest_out", model.eq.c_out),
                         ("classifier", model.basic.classifier)):
        weights[name] = (module.weight, "real", 0)
    with torch.no_grad():
        for name, (weight, kind, axis) in weights.items():
            quantized = (quant_complex_weight(weight, 8, axis) if kind == "complex"
                         else quant_complex_vector(weight, 8) if kind == "vector"
                         else quant_real_weight(weight, 8))
            diff = (quantized - weight).abs()
            result[name] = {"relative_rmse": float(diff.norm() / weight.abs().norm().clamp_min(1e-30)),
                            "max_absolute_error": float(diff.max()),
                            "real_scalar_entries": weight.numel() * (2 if weight.is_complex() else 1)}
    return result


def run_condition(model, ref, cfg, channel, args, seed, snr, target):
    point_seed = args.eval_seed + seed + int(snr * 10)
    torch.manual_seed(point_seed)
    tx = OfdmTxTorch(cfg, torch.device("cpu"))
    with torch.no_grad():
        transmitted, labels, _ = tx.transmit(args.slots)
        received, taps = apply_fading_awgn(
            channel, transmitted, torch.full((args.slots,), float(snr)))
        residual = noise_residual(transmitted, received, taps, channel)
    stimulus = {"transmitted": sha256_tensor(transmitted), "labels": sha256_tensor(labels),
                "taps": sha256_tensor(taps), "noise_residual": sha256_tensor(residual),
                "received": sha256_tensor(received)}
    truth = labels.to(torch.bool)
    predictions, margins, local, stage = {}, {}, {}, {}
    started = time.perf_counter()
    with torch.inference_mode():
        # Exact agreement with untouched shared model is a release gate.
        original = StructuredQDCCN(cfg, rank=0, eq_dense=True, slim=True, qc=QConfig())
        original.load_state_dict(model.state_dict())
        original.eval()
        model.set_variant("w8a8_matrix")
        agreement = {}
        for backend in ("matrix", "fft"):
            model.set_variant("w8a8_" + backend)
            original.set_transform_backend(backend)
            a = model(received[:min(args.batch, args.slots)])
            b = original(received[:min(args.batch, args.slots)])
            agreement[backend] = float((a - b).abs().max())
            if agreement[backend] != 0.0:
                raise AssertionError(f"{backend} forward differs from original: {agreement[backend]}")
        del original
        for variant in args.variants:
            if variant != "structured_fp32":
                model.set_variant(variant)
            pp, mm = [], []
            trace = variant in ("w8a8_matrix", "qat_weights_all_float")
            model.trace_enabled = trace
            for start in range(0, args.slots, args.batch):
                yy = received[start:start + args.batch]
                out = ref(yy) if variant == "structured_fp32" else model(yy)
                pp.append(out[..., 1] > .5)
                mm.append((out[..., 1] - out[..., 0]).abs().reshape(len(yy), -1).amin(1))
                if trace:
                    if variant == "w8a8_matrix":
                        for name, value in model.local.items():
                            local.setdefault(name, []).append(value.cpu().numpy())
                    stage.setdefault(variant, {})[start] = {
                        name: value.cpu().numpy() for name, value in model.stages.items()}
            predictions[variant] = torch.cat(pp)
            margins[variant] = torch.cat(mm).cpu().numpy()
    baseline = predictions["w8a8_matrix"]
    baseline_mask = baseline != truth
    arrays = {"baseline_slot_errors": baseline_mask.reshape(args.slots, -1).sum(1).numpy()}
    bits_per_slot = cfg.D_slot * cfg.m
    summaries = {}
    for variant, prediction in predictions.items():
        errors = prediction != truth
        slot_err = errors.reshape(args.slots, -1).sum(1).numpy()
        baseline_only = (baseline_mask & ~errors).reshape(args.slots, -1).sum(1).numpy()
        variant_only = (~baseline_mask & errors).reshape(args.slots, -1).sum(1).numpy()
        flips = (prediction != baseline).reshape(args.slots, -1).sum(1).numpy()
        assert np.array_equal(flips, baseline_only + variant_only)
        difference = (slot_err - arrays["baseline_slot_errors"]) / bits_per_slot
        arrays[variant + "__slot_errors"] = slot_err
        arrays[variant + "__baseline_only_errors"] = baseline_only
        arrays[variant + "__variant_only_errors"] = variant_only
        arrays[variant + "__flips"] = flips
        arrays[variant + "__min_abs_probability_margin"] = margins[variant]
        stats = (summarize_paired_slots(slot_err, arrays["baseline_slot_errors"],
                                       bits_per_slot, bootstrap_reps=args.bootstrap,
                                       bootstrap_seed=args.bootstrap_seed + point_seed)
                 if variant in PRIMARY_CONTRASTS and args.bootstrap else None)
        summaries[variant] = {"errors": int(slot_err.sum()), "ber": float(slot_err.mean() / bits_per_slot),
                              "delta_ber_vs_w8a8": float(difference.mean()),
                              "paired_slot_statistics": stats,
                              "changed_decisions": int(flips.sum()),
                              "baseline_only_errors": int(baseline_only.sum()),
                              "variant_only_errors": int(variant_only.sum())}
    local_summary = {}
    for name, values in local.items():
        arr = np.concatenate(values)
        arrays["local__" + name] = arr
        aggregate = arr.sum(0)
        local_summary[name] = {"relative_rmse": float(np.sqrt(aggregate[0] / max(aggregate[1], 1e-30))),
                               "max_absolute_error": float(np.sqrt(arr[:, 2].max())),
                               "out_of_range_values": int(aggregate[3]),
                               "values": int(aggregate[4]),
                               "out_of_range_fraction": float(aggregate[3] / aggregate[4])}
    if "qat_weights_all_float" in stage:
        for name in stage["w8a8_matrix"][0]:
            values = []
            for start, baseline_stage in stage["w8a8_matrix"].items():
                actual = torch.from_numpy(baseline_stage[name])
                oracle = torch.from_numpy(stage["qat_weights_all_float"][start][name])
                values.append(metrics(actual, oracle).numpy())
            arrays["propagated_stage__" + name] = np.concatenate(values)
    assert stimulus["received"] == sha256_tensor(received)
    assert stimulus["labels"] == sha256_tensor(labels)
    assert stimulus["taps"] == sha256_tensor(taps)
    np.savez_compressed(target.with_suffix(".npz"), **arrays)
    record = {"status": "EXPLORATORY_FIXED_QAT_WEIGHT_SLOT_PAIRED_PRECISION_AUDIT",
              "training_seed": seed, "config": cfg.summary(), "tag": args.active_tag,
              "channel": channel.model, "snr_db": snr, "slots": args.slots,
              "bits": args.slots * bits_per_slot, "point_seed": point_seed,
              "pdp_mode": channel.pdp_mode, "stimulus_sha256": stimulus,
              "baseline_exact_original_max_absolute_diff": agreement,
              "variants": summaries, "local_operator_metrics_baseline": local_summary,
              "slot_data": target.with_suffix(".npz").name,
              "slot_data_sha256": file_hash(target.with_suffix(".npz")),
              "elapsed_seconds": time.perf_counter() - started,
              "limitations": ["Restorations retain QAT learned weights, not FP32 training weights.",
                              "Bootstrap intervals condition on this checkpoint and simulation distribution.",
                              "All comparisons are exploratory and no multiplicity-adjusted hypothesis tests are claimed.",
                              "Higher activation precision retains frozen calibration maxima; no recalibration or retraining.",
                              "QPSK is software-only; the current hardware image is BPSK long-CP."]}
    save_json(target, record)
    print(f"{args.active_tag} seed={seed} {channel.model} {snr:g}dB: "
          f"{args.slots} slots, {len(args.variants)} variants, "
          f"{record['elapsed_seconds']:.1f}s; baseline BER="
          f"{summaries['w8a8_matrix']['ber']:.6f}", flush=True)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tags", nargs="+", default=["b1_longcp", "b2_longcp", "b2_shortcp"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--channels", nargs="+", default=["ETU", "EVA"])
    parser.add_argument("--snrs", nargs="+", type=float, default=[12, 16, 20, 24])
    parser.add_argument("--slots", type=int, default=2000)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS))
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--bootstrap-seed", type=int, default=20261001)
    parser.add_argument("--eval-seed", type=int, default=20261001)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if min(args.slots, args.batch, args.threads) < 1 or "w8a8_matrix" not in args.variants:
        parser.error("positive slots/batch/threads and w8a8_matrix baseline required")
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        parser.error(f"Output already exists: {output}; choose a new run directory.")
    manifest = {"status": "SMOKE_NOT_FOR_PUBLICATION" if args.smoke else "EXPLORATORY_PRECISION_SCREEN",
                "python": platform.python_version(), "python_executable": sys.executable,
                "torch": torch.__version__, "numpy": np.__version__, "device": "cpu",
                "threads": args.threads, "args": {key: str(value) if isinstance(value, Path) else value
                                                  for key, value in vars(args).items()},
                "source_hashes": {name: file_hash(HERE / name) for name in (
                    "run_precision_localization_pdpfix.py", "struct_quant_model.py", "quant.py",
                    "quant_model.py", "struct_model.py", "config.py", "ofdm.py", "train.py",
                    "paired_research_stats.py")},
                "scope": {
                    "block_restore": "Disable the block's weight/input/output activation quantization; retain external boundary quantizers.",
                    "basic_gain_restore": "Use unquantized gain and bypass its following q_out; keep normalization and FFT quantizers.",
                    "nonlinear_restore": "Exact RMS, reciprocal division and torch.tanh together; retain W8A8 elsewhere.",
                    "mixed_precision_probes": "W8A16 and W16A8 reinterpret the fixed QAT checkpoint/frozen maxima without retraining or recalibration.",
                    "float_qat_counterfactual": "Disable quantization and nonlinear approximations while retaining learned QAT weights and learned delta.",
                    "matrix_fft": "Fixed W8A8 checkpoint, same calibration and traffic; transform backend only is changed.",
                    "planned_slots_per_condition": 2000,
                    "training_seeds_vs_slots": "Three trained seeds are replication; slots quantify conditional simulation uncertainty only.",
                    "bootstrap_contrasts": sorted(PRIMARY_CONTRASTS)},
                "conditions": [], "checkpoints": {}}
    save_json(manifest_path, manifest)
    for tag in args.tags:
        args.active_tag = tag
        cfg = OfdmConfig(nbits=int(tag[1]), long_cp=tag.endswith("longcp"))
        for seed in args.seeds:
            suffix = f"_pdpfix_seed{seed}.pt"
            qat_path = HERE / "ckpt" / f"sq_{tag}_s_r0_slim_w8a8{suffix}"
            fp_path = HERE / "ckpt" / f"struct_{tag}_s_r0_slim{suffix}"
            state = torch.load(qat_path, map_location="cpu", weights_only=True)
            model = DiagnosticReceiver(cfg, rank=0, eq_dense=True, slim=True, qc=QConfig())
            model.load_state_dict(state)
            model.eval()
            ref = StructuredDCCN(cfg, rank=0, eq_dense=True, slim=True)
            ref.load_state_dict(torch.load(fp_path, map_location="cpu", weights_only=True))
            ref.eval()
            key = f"{tag}_seed{seed}"
            manifest["checkpoints"][key] = {
                "qat": str(qat_path), "qat_sha256": file_hash(qat_path),
                "structured_fp32": str(fp_path), "structured_fp32_sha256": file_hash(fp_path),
                "qat_delta": float(model.eq.delta.detach().abs()),
                "static_weight_quantization_metrics": static_weight_metrics(model),
                "frozen_activation_maxima": {name: float(module.amax) for name, module in model.named_modules()
                                             if isinstance(module, ActQ)},
                "uncalibrated_quantizers": [name for name, module in model.named_modules()
                                            if isinstance(module, ActQ) and not bool(module.calibrated)]}
            if manifest["checkpoints"][key]["uncalibrated_quantizers"]:
                raise AssertionError("Frozen calibration buffer missing")
            for channel_name in args.channels:
                channel = ChannelTorch(cfg, torch.device("cpu"), model=channel_name)
                assert channel.pdp_mode == "3gpp_db_to_linear_power"
                for snr in args.snrs:
                    target = output / f"{tag}_seed{seed}_{channel_name}_{snr:g}dB.json"
                    run_condition(model, ref, cfg, channel, args, seed, snr, target)
                    manifest["conditions"].append(target.name)
                    save_json(manifest_path, manifest)
    manifest["completion"] = "COMPLETE"
    save_json(manifest_path, manifest)


if __name__ == "__main__":
    main()
