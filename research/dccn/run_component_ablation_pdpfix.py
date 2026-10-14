"""Retrained, common-stimulus component ablation of the corrected-PDP receiver.

Only new outputs are written.  All arms inherit the same seed-specific dense
teacher, then receive exactly the same online batch at every fixed-budget
update.  The final checkpoint is used; no test-set or training-BER selection.
This is an independent common-budget fine-tuning study, not a comparison with
the differently trained mainline checkpoints and not post-hoc module knockout.
"""

import argparse
import hashlib
import json
import math
import platform
import statistics
import time
from pathlib import Path

import numpy as np
import torch

from config import OfdmConfig
from layers import rms_norm
from ofdm import ChannelTorch, OfdmTxTorch
from paired_research_stats import summarize_paired_slots
from receiver import ce_loss
from struct_model import StructuredDCCN
from train import apply_fading_awgn, sample_mixed_snr


HERE = Path(__file__).resolve().parent
ROOT = HERE / "experiments" / "component_ablation_20261001"
CKPT = HERE / "ckpt"
TRAIN_CHANNELS = ("flat", "EPA", "EVA", "ETU")
TEST_CHANNELS = ("EVA", "ETU")
BASE = "slim2"
VARIANTS = {
    "slim2": "CP folding, learned front/back mixes and RE gains, differential depth 2",
    "discard_cp": "Zero CP rows in both analysis transforms; retain inverse CP synthesis and full-slot RMS; reproject teacher using the changed analysis operator",
    "fixed_front": "Replace front frequency mixing by fixed identity and its bias by zero; fine-tune every remaining component",
    "fixed_back": "Replace back correction mixing by fixed identity and its bias by zero; fine-tune every remaining component",
    "fixed_basic_gain": "Fix data-RE gain to one and bias to zero; fine-tune every remaining component",
    "serial2": "Keep two tanh blocks and readout, use serial h_j=tanh(W_j h_(j-1)+b_j) instead of differential chaining",
    "full6": "Differential depth 6, with every other component unchanged",
}


def file_hash(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_hash(tensor):
    array = tensor.detach().contiguous().cpu().numpy()
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()


def save_new(path, payload):
    with open(path, "x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def parameter_count(model, trainable_only=False):
    return sum(parameter.numel() * (2 if parameter.is_complex() else 1)
               for parameter in model.parameters()
               if not trainable_only or parameter.requires_grad)


def operation_accounting(model):
    cfg = model.cfg
    f, n, s = cfg.N_SYM, cfg.K, cfg.S
    depth, width = len(model.eq.c_res), model.eq.c_in.out_features
    return {
        "transform_only_fft_backbone_complex_mac_proxy": model.accounting()["macs_total"],
        "transform_only_direct_matrix_reference_complex_macs": 3 * f * s * n + 2 * f * n * n + cfg.D_slot,
        "channel_subnet_real_dense_macs": f * n * 2 * width + depth * width * width + depth * width * f * n * 2,
        "classifier_real_dense_macs": cfg.D_slot * 4 * 2 * cfg.m,
        "analysis_cp_fold_complex_additions_in_fft_mapping": 0 if model.variant == "discard_cp" else 2 * f * cfg.CP,
        "fixed_identity_maps_are_still_dense_gemm_in_this_reference_graph": model.variant in ("fixed_front", "fixed_back"),
        "scope": "Separate complex and real MAC categories, not total FLOPs. FFT proxy adopts shared accounting convention; Python uses direct matrices and keeps literal identity GEMM / unit gain arithmetic. CP-fold addition count is an algebraic mapping count, not measured CPU work. Excludes RMS, nonlinearities, division, memory traffic and control; not an HLS resource/latency prediction.",
    }


class ComponentDCCN(StructuredDCCN):
    """New experiment-local graph; shared model implementations stay untouched."""

    def __init__(self, cfg, variant, basic_state, dense_state):
        if variant not in VARIANTS:
            raise ValueError(variant)
        super().__init__(cfg, rank=0, eq_dense=True, slim=variant != "full6")
        self.variant = variant
        if variant == "discard_cp":
            with torch.no_grad():
                self.eq.Fd[:cfg.CP].zero_()
                self.basic.Fd[:cfg.CP].zero_()
        # Reprojection is essential: a CP-discard arm must not retain the gain
        # coefficients projected onto a different CP-folding basis.
        self.init_from_float(basic_state, dense_state)
        with torch.no_grad():
            if variant in ("fixed_front", "fixed_back"):
                mix = self.eq.m_front if variant == "fixed_front" else self.eq.m_back
                bias = self.eq.b_front if variant == "fixed_front" else self.eq.b_back
                mix.weight.copy_(torch.eye(cfg.K, dtype=mix.weight.dtype))
                mix.weight.requires_grad_(False)
                bias.zero_()
                bias.requires_grad_(False)
            elif variant == "fixed_basic_gain":
                self.basic.g.fill_(1)
                self.basic.bias.zero_()
                self.basic.g.requires_grad_(False)
                self.basic.bias.requires_grad_(False)

    def equalize(self, y):
        cfg, eq = self.cfg, self.eq
        y = rms_norm(y)
        batch = y.shape[0]
        yf = torch.einsum("bfs,sn->bfn", y, eq.Fd.to(y.dtype))
        frequency = eq.m_front(yf) + eq.b_front.to(y.dtype)
        q = torch.cat((frequency.real, frequency.imag), dim=-1).reshape(batch, -1)
        h = eq.c_in(q)
        x, states = h, []
        for linear in eq.c_res:
            states.append(torch.tanh(linear(x)))
            if self.variant == "serial2":
                x = states[-1]
            else:
                x = (states[-2] if len(states) >= 2 else h) - states[-1]
        readout = eq.c_out(torch.cat(states, dim=-1)).view(batch, cfg.N_SYM, cfg.K, 2)
        channel = torch.complex(readout[..., 0], readout[..., 1])
        equalized = frequency * channel.conj() / (channel.abs().square() + 0.1)
        restored = eq.m_back(equalized) @ eq.FdcT.to(y.dtype) + eq.b_back.to(y.dtype)
        return restored

    def forward(self, y):
        return self.basic(self.equalize(y))


def graph_checks(cfg, sources, device):
    """Verify control equivalence and intervention semantics before training."""
    basic_state, dense_state = sources
    torch.manual_seed(81001)
    reference = StructuredDCCN(cfg, rank=0, eq_dense=True, slim=True)
    reference.init_from_float(basic_state, dense_state)
    local = ComponentDCCN(cfg, BASE, basic_state, dense_state)
    y = torch.complex(torch.randn(3, cfg.N_SYM, cfg.S),
                      torch.randn(3, cfg.N_SYM, cfg.S))
    with torch.no_grad():
        delta = (reference(y) - local(y)).abs().max().item()
    if delta != 0.0:
        raise AssertionError(f"Experiment control graph differs from shared model: {delta}")
    discard = ComponentDCCN(cfg, "discard_cp", basic_state, dense_state)
    if int(torch.count_nonzero(discard.eq.Fd[:cfg.CP])):
        raise AssertionError("CP-discard front transform still uses CP")
    if int(torch.count_nonzero(discard.basic.Fd[:cfg.CP])):
        raise AssertionError("CP-discard basic transform still uses CP")
    if not torch.equal(discard.eq.FdcT, reference.eq.FdcT):
        raise AssertionError("CP-discard intervention changed inverse synthesis")
    operator_delta = 0.0
    with torch.no_grad():
        full = torch.einsum("bfs,sn->bfn", y, discard.eq.Fd)
        plain = torch.einsum("bfs,sn->bfn", y[:, :, cfg.CP:],
                            discard.eq.Fd[cfg.CP:])
        operator_delta = (full - plain).abs().max().item()
    if operator_delta > 2e-6:
        raise AssertionError("CP-discard transform does not match body-only DFT")
    return {"control_vs_shared_max_probability_error": delta,
            "discard_vs_body_only_max_transform_error": operator_delta,
            "inverse_synthesis_unchanged": True,
            "device": str(device)}


def make_training_batch(cfg, tx, channels, seed, step, batch, device):
    torch.manual_seed(10_000_000 + 100_000 * seed + step)
    transmitted, labels, _ = tx.transmit(batch)
    snr = sample_mixed_snr(batch, device)
    received = torch.empty_like(transmitted)
    for index, name in enumerate(TRAIN_CHANNELS):
        selection = torch.arange(index, batch, len(TRAIN_CHANNELS), device=device)
        received[selection], _ = apply_fading_awgn(
            channels[name], transmitted[selection], snr[selection])
    return received, labels, snr


def train_seed(cfg, seed, sources, args, output, device):
    models, optimizers, schedulers, metadata = {}, {}, {}, {}
    for variant in args.variants:
        torch.manual_seed(seed)
        model = ComponentDCCN(cfg, variant, *sources).to(device).train()
        models[variant] = model
        optimizers[variant] = torch.optim.Adam(
            [p for p in model.parameters() if p.requires_grad],
            lr=args.lr, weight_decay=args.weight_decay)
        schedulers[variant] = torch.optim.lr_scheduler.StepLR(
            optimizers[variant], step_size=500, gamma=0.95)
        metadata[variant] = {
            "description": VARIANTS[variant],
            "total_parameter_real_scalar_equivalents": parameter_count(model),
            "trainable_parameter_real_scalar_equivalents": parameter_count(model, True),
            "channel_estimator_depth": len(model.eq.c_res),
            "analysis_operator_sha256": tensor_hash(model.eq.Fd),
            "basic_analysis_operator_sha256": tensor_hash(model.basic.Fd),
            "inverse_operator_sha256": tensor_hash(model.eq.FdcT),
            "operation_accounting": operation_accounting(model),
        }
    tx = OfdmTxTorch(cfg, device)
    channels = {name: ChannelTorch(cfg, device, model=name) for name in TRAIN_CHANNELS}
    if any(channel.pdp_mode != "3gpp_db_to_linear_power"
           for name, channel in channels.items() if name != "flat"):
        raise AssertionError("Corrected PDP is required")
    chain = hashlib.sha256()
    start = time.monotonic()
    training_points = []
    for step in range(1, args.iters + 1):
        received, labels, snr = make_training_batch(
            cfg, tx, channels, seed, step, args.batch, device)
        hashes = {"received": tensor_hash(received), "labels": tensor_hash(labels),
                  "snr": tensor_hash(snr)}
        chain.update(str(step).encode("ascii"))
        chain.update(json.dumps(hashes, sort_keys=True).encode("ascii"))
        losses = {}
        for variant, model in models.items():
            optimizer = optimizers[variant]
            optimizer.zero_grad(set_to_none=True)
            soft = model(received)
            loss = ce_loss(soft, labels)
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError(f"Nonfinite loss: seed={seed}, {variant}, step={step}")
            loss.backward()
            optimizer.step()
            schedulers[variant].step()
            losses[variant] = float(loss.detach())
        # All arms consume the same tensor, not merely equal RNG seeds.
        if tensor_hash(received) != hashes["received"] or tensor_hash(labels) != hashes["labels"]:
            raise AssertionError("Training input mutated by an arm")
        if step == 1 or step % args.log_every == 0 or step == args.iters:
            training_points.append({"step": step, "loss": losses,
                                    "stimulus_sha256": hashes,
                                    "elapsed_seconds": time.monotonic() - start})
            print(f"seed={seed} step={step}/{args.iters} "
                  + " ".join(f"{v}={losses[v]:.5f}" for v in args.variants)
                  + f" elapsed={time.monotonic()-start:.1f}s", flush=True)
    seconds = time.monotonic() - start
    for variant, model in models.items():
        path = output / f"{variant}_seed{seed}_final.pt"
        if path.exists():
            raise FileExistsError(path)
        model.eval()
        torch.save(model.state_dict(), path)
        metadata[variant].update({"checkpoint": str(path), "sha256": file_hash(path),
                                  "updates": args.iters,
                                  "examples_seen": args.iters * args.batch,
                                  "training_stimulus_chain_sha256": chain.hexdigest()})
    return models, {"arms": metadata, "training_points": training_points,
                    "all_arm_training_seconds": seconds,
                    "training_stimulus_chain_sha256": chain.hexdigest()}


def error_masks(model, received, labels, batch):
    mask = torch.empty(labels.shape, dtype=torch.bool, device=labels.device)
    with torch.no_grad():
        for begin in range(0, labels.shape[0], batch):
            end = min(begin + batch, labels.shape[0])
            soft = model(received[begin:end])
            mask[begin:end] = (soft[..., 1] > 0.5) != labels[begin:end]
    return mask


def paired_record(variant_mask, base_mask, bootstrap_seed):
    totals = {
        "variant_only_error": int((variant_mask & ~base_mask).sum()),
        "base_only_error": int((~variant_mask & base_mask).sum()),
        "both_error": int((variant_mask & base_mask).sum()),
        "neither_error": int((~variant_mask & ~base_mask).sum()),
    }
    if sum(totals.values()) != base_mask.numel():
        raise AssertionError("Paired bit counts do not sum to total bits")
    variant_slot = variant_mask.flatten(1).sum(1).cpu().tolist()
    base_slot = base_mask.flatten(1).sum(1).cpu().tolist()
    per_slot_delta = [a - b for a, b in zip(variant_slot, base_slot)]
    slot_summary = summarize_paired_slots(
        variant_slot, base_slot, base_mask[0].numel(),
        bootstrap_reps=2000, bootstrap_seed=bootstrap_seed)
    return {"bit_contingency": totals,
            "variant_minus_base_ber": (totals["variant_only_error"]
                                        - totals["base_only_error"]) / base_mask.numel(),
            "per_slot_error_count_delta": per_slot_delta,
            "slot_cluster_descriptive_summary": slot_summary,
            "within_slot_bits_are_not_independent_replicates": True}


def threshold(snrs, errors, bits, target):
    raw = np.asarray(errors, dtype=float) / bits
    monotonic = np.minimum.accumulate(raw)
    if monotonic[0] <= target:
        return {"db": None, "censoring": "left_of_grid", "bound_db": snrs[0]}
    if monotonic[-1] > target:
        return {"db": None, "censoring": "right_of_grid", "bound_db": snrs[-1]}
    for index in range(len(snrs) - 1):
        if monotonic[index] > target >= monotonic[index + 1]:
            # A zero count gives an upper bound, not a measured zero probability.
            # The half-count floor affects interpolation only if its bracket is zero.
            left = max(monotonic[index], 0.5 / bits)
            right = max(monotonic[index + 1], 0.5 / bits)
            weight = (math.log10(target) - math.log10(left)) / (math.log10(right) - math.log10(left))
            return {"db": float(snrs[index] + weight * (snrs[index + 1] - snrs[index])),
                    "censoring": None, "bracket_db": [snrs[index], snrs[index + 1]],
                    "zero_bracket_half_count_floor_used": bool(monotonic[index + 1] == 0)}
    raise AssertionError("Threshold crossing not found")


def evaluate_seed(cfg, seed, models, args, device):
    tx = OfdmTxTorch(cfg, device)
    results = {}
    for name in TEST_CHANNELS:
        channel = ChannelTorch(cfg, device, model=name)
        if channel.pdp_mode != "3gpp_db_to_linear_power":
            raise AssertionError(channel.pdp_mode)
        points = []
        for snr_db in args.snrs:
            point_seed = 20260904 + seed + int(snr_db * 10)
            torch.manual_seed(point_seed)
            with torch.no_grad():
                transmitted, labels, _ = tx.transmit(args.slots)
                snr = torch.full((args.slots,), snr_db, device=device)
                received, taps = apply_fading_awgn(channel, transmitted, snr)
            hashes = {"transmitted": tensor_hash(transmitted), "labels": tensor_hash(labels),
                      "received": tensor_hash(received), "taps": tensor_hash(taps)}
            masks = {variant: error_masks(model, received, labels, args.eval_batch)
                     for variant, model in models.items()}
            if tensor_hash(received) != hashes["received"] or tensor_hash(labels) != hashes["labels"]:
                raise AssertionError("Evaluation stimulus mutated by an arm")
            errors = {variant: int(mask.sum()) for variant, mask in masks.items()}
            paired = {variant: paired_record(mask, masks[BASE],
                                             20261001 + 1000 * seed + int(snr_db * 10))
                      for variant, mask in masks.items() if variant != BASE}
            bits = int(labels.numel())
            point = {"snr_db": snr_db, "point_seed": point_seed, "slots": args.slots,
                     "bits": bits, "errors": errors,
                     "ber": {variant: count / bits for variant, count in errors.items()},
                     "stimulus_sha256": hashes, "paired_vs_slim2": paired}
            point["per_slot_error_counts"] = {
                variant: mask.flatten(1).sum(1).cpu().tolist()
                for variant, mask in masks.items()}
            points.append(point)
            print(f"seed={seed} {name} {snr_db:g}dB "
                  + " ".join(f"{v}={point['ber'][v]:.5f}" for v in args.variants), flush=True)
        thresholds = {variant: threshold(args.snrs, [p["errors"][variant] for p in points],
                                          points[0]["bits"], args.target_ber)
                      for variant in models}
        results[name] = {"points": points, "thresholds": thresholds,
                         "pdp_db": channel.pdp_db.cpu().tolist(),
                         "normalized_linear_pdp": channel.pdp.cpu().tolist(),
                         "resampling_operator_sha256": tensor_hash(channel.A)}
    return results


def finite_stats(values):
    return {"per_seed": values,
            "mean": statistics.mean(values) if all(v is not None for v in values) else None,
            "sample_sd": statistics.stdev(values) if len(values) >= 2
                         and all(v is not None for v in values) else None}


def aggregate(records, args):
    channels = {}
    for channel in TEST_CHANNELS:
        arms = {}
        for variant in args.variants:
            thresholds = [record["evaluation"][channel]["thresholds"][variant]["db"]
                          for record in records]
            base = [record["evaluation"][channel]["thresholds"][BASE]["db"]
                    for record in records]
            deltas = [None if v is None or b is None else v - b for v, b in zip(thresholds, base)]
            at_snrs = {}
            for point_index, snr in enumerate(args.snrs):
                bers = [record["evaluation"][channel]["points"][point_index]["ber"][variant]
                        for record in records]
                paired = [record["evaluation"][channel]["points"][point_index]["ber"][variant]
                          - record["evaluation"][channel]["points"][point_index]["ber"][BASE]
                          for record in records]
                at_snrs[str(snr)] = {"ber": finite_stats(bers),
                                     "paired_variant_minus_slim2_ber": finite_stats(paired)}
            arms[variant] = {"snr_at_ber_target_db": finite_stats(thresholds),
                             "paired_variant_minus_slim2_db": finite_stats(deltas),
                             "at_snr": at_snrs}
        channels[channel] = arms
    return {"training_seeds": args.seeds, "n_independent_training_runs_per_arm": len(records),
            "target_ber": args.target_ber, "channels": channels,
            "inference_boundary": "Descriptive independent training-seed mean/sample SD, no p-values or population claims; per-slot paired counts retained for cluster-aware follow-up. Mainline training budget differs."}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--variants", default=",".join(VARIANTS))
    parser.add_argument("--seeds", default=None)
    parser.add_argument("--iters", type=int, default=None)
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--slots", type=int, default=None)
    parser.add_argument("--snrs", default=None)
    parser.add_argument("--eval-batch", type=int, default=250)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--log-every", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--target-ber", type=float, default=0.02)
    args = parser.parse_args()
    args.variants = args.variants.split(",")
    args.seeds = [int(x) for x in (args.seeds or ("1" if args.smoke else "1,2,3")).split(",")]
    args.iters = args.iters if args.iters is not None else (20 if args.smoke else 1000)
    args.batch = args.batch if args.batch is not None else (32 if args.smoke else 256)
    args.slots = args.slots if args.slots is not None else (16 if args.smoke else 2000)
    args.snrs = [float(x) for x in (args.snrs or ("12,20" if args.smoke else "0,4,8,12,16,20,24,28")).split(",")]
    args.out = args.out or ROOT / ("smoke" if args.smoke else "common_budget_7arm")
    if BASE not in args.variants or len(set(args.variants)) != len(args.variants):
        parser.error("Unique arms must include slim2")
    if any(v not in VARIANTS for v in args.variants):
        parser.error("Unknown variant")
    if not args.seeds or min(args.seeds) < 1 or len(set(args.seeds)) != len(args.seeds):
        parser.error("Seeds must be unique positive integers")
    if min(args.iters, args.batch, args.slots, args.eval_batch, args.threads, args.log_every) <= 0:
        parser.error("All counts must be positive")
    if len(args.snrs) < 2 or args.snrs != sorted(set(args.snrs)):
        parser.error("At least two strictly increasing SNR values are required")
    if not 0 < args.target_ber < 1 or args.lr <= 0 or args.weight_decay < 0:
        parser.error("Invalid optimization/threshold settings")
    # A changed protocol is allowed only as an explicitly marked smoke/pilot.
    if not args.smoke and (args.seeds != [1, 2, 3] or args.iters != 1000
                           or args.batch != 256 or args.slots != 2000
                           or args.snrs != [0., 4., 8., 12., 16., 20., 24., 28.]
                           or args.variants != list(VARIANTS) or args.target_ber != 0.02):
        parser.error("Common-budget protocol is fixed; use --smoke for a modified pilot")
    return args


def main():
    args = parse_args()
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    device = torch.device("cpu")
    cfg = OfdmConfig(nbits=1, long_cp=True)
    sources = {}
    for seed in args.seeds:
        paths = (CKPT / f"basic_b1_longcp_pdpfix_seed{seed}.pt",
                 CKPT / f"dccn_b1_longcp_pdpfix_seed{seed}.pt")
        if any(not path.is_file() for path in paths):
            raise FileNotFoundError(paths)
        sources[seed] = paths
    if args.out.exists():
        raise FileExistsError(f"Never overwrite a prior experiment: {args.out}")
    args.out.mkdir(parents=True)
    status = "SMOKE_NOT_FOR_PUBLICATION" if args.smoke else "COMMON_BUDGET_RETRAINED_COMPONENT_ABLATION"
    protocol = {
        "status": status, "system": cfg.summary(), "variants": VARIANTS,
        "active_variants": args.variants, "seeds": args.seeds,
        "training": {"updates": args.iters, "batch": args.batch,
                     "examples_per_arm_seed": args.iters * args.batch,
                     "teacher": "same training-seed dense corrected-PDP checkpoints for every arm",
                     "initialization": "least-squares projection onto each arm's own analysis operator; prescribed identity/zero values for removed corrections",
                     "checkpoint_selection": "final fixed-budget checkpoint; no early stopping",
                     "paired_data": "one transmit/fading/noise draw per update, physically identical received tensor consumed by all arms",
                     "data_seed_rule": "10000000 + 100000*training_seed + step",
                     "all_step_stimulus_hash_chain": True,
                     "snr_law": "dccn.train.sample_mixed_snr",
                     "channels": TRAIN_CHANNELS, "lr": args.lr,
                     "weight_decay": args.weight_decay, "optimizer": "Adam",
                     "scheduler": "StepLR(gamma=0.95, step_size=500)"},
        "evaluation": {"channels": TEST_CHANNELS, "snrs_db": args.snrs,
                       "slots_per_channel_snr_seed": args.slots,
                       "bits_per_slot": cfg.D_slot * cfg.m, "target_ber": args.target_ber,
                       "shared_stimulus": "one batch generation per condition, all arms use identical inputs",
                       "point_seed_rule": "20260904 + training_seed + int(10*SNR_dB)",
                       "threshold": "first crossing of nonincreasing BER envelope, log10(BER)-linear interpolation; explicit censoring; half-count interpolation floor only if crossing bracket has zero errors",
                       "uncertainty": "independent training-seed sample SD; bit counts not treated as independent replicates"},
        "boundary": "A reproject-and-fine-tune architectural intervention, not from-scratch training. Equal update/example budgets, not equal parameter counts or wall time. 1000x256 budget differs from mainline up-to-3000x512 plus BER checkpoint selection. No JTAG, latency or resource conclusions.",
        "environment": {"python": platform.python_version(), "torch": torch.__version__,
                        "numpy": np.__version__, "device": str(device), "threads": args.threads},
        "source_files_sha256": {str(path): file_hash(path) for path in
                                (Path(__file__), HERE / "struct_model.py", HERE / "ofdm.py",
                                 HERE / "train.py", HERE / "config.py", HERE / "layers.py")},
        "source_checkpoints": {str(seed): {str(path): file_hash(path) for path in paths}
                               for seed, paths in sources.items()},
    }
    save_new(args.out / "protocol.json", protocol)
    all_records = []
    start = time.monotonic()
    for seed in args.seeds:
        state = tuple(torch.load(path, map_location="cpu", weights_only=True)
                      for path in sources[seed])
        checks = graph_checks(cfg, state, device)
        models, training = train_seed(cfg, seed, state, args, args.out, device)
        evaluation = evaluate_seed(cfg, seed, models, args, device)
        record = {"status": status, "seed": seed, "graph_checks": checks,
                  "training": training, "evaluation": evaluation}
        save_new(args.out / f"seed{seed}.json", record)
        all_records.append(record)
        del models
    summary = {"status": status, "protocol_sha256": file_hash(args.out / "protocol.json"),
               "seed_files": {str(seed): {"path": str(args.out / f"seed{seed}.json"),
                                            "sha256": file_hash(args.out / f"seed{seed}.json")}
                              for seed in args.seeds},
               "elapsed_seconds": time.monotonic() - start,
               "summary": aggregate(all_records, args)}
    save_new(args.out / "summary.json", summary)
    # Teacher files are immutable evidence; check for concurrent modification.
    for paths in protocol["source_checkpoints"].values():
        for path, digest in paths.items():
            if file_hash(path) != digest:
                raise AssertionError(f"Teacher checkpoint changed: {path}")
    print(f"Completed {status}: {args.out / 'summary.json'} ({summary['elapsed_seconds']:.1f}s)", flush=True)


if __name__ == "__main__":
    main()
