"""Paired deployment-transfer audit, frozen local-A12 and rev-T candidates."""
import argparse
import ctypes as ct
import hashlib
import json
import platform
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "dccn"))
from config import OfdmConfig
from eval_paired_b1_longcp_pdpfix import noise_residual, sha256_tensor
from ofdm import ChannelTorch, OfdmTxTorch
from paired_research_stats import summarize_paired_slots
from quant import QConfig
from run_back_precision_followup import BackPrecisionReceiver, point_seed
from struct_quant_model import StructuredQDCCN
from train import apply_fading_awgn

BASE = HERE.parent / "pdpfix_opt_20260928"
PRIOR = ROOT / "dccn/experiments/back_precision_followup_20261002/confirmation_2000"
CKPT = ROOT / "dccn/ckpt/sq_b1_longcp_s_r0_slim_w8a8_pdpfix_seed1.pt"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def array(name, path=HERE / "stim.h", dtype=np.float32):
    match = re.search(r"static const \w+ " + name + r"\[\d+\] = \{(.*?)\};",
                      path.read_text(encoding="utf-8"), re.S)
    if not match:
        raise ValueError(name)
    return np.fromstring(match.group(1).replace(",", " "), sep=" ", dtype=dtype)


def library(name):
    lib = ct.CDLL(str(HERE / (name + ".dll")))
    lib.run_slots.argtypes = [ct.c_int, ct.c_void_p, ct.c_void_p, ct.c_void_p, ct.c_void_p]
    lib.run_slots.restype = ct.c_int
    return lib


def native(lib, received):
    yr = np.ascontiguousarray(received.real.numpy().reshape(-1, 560), dtype=np.float32)
    yi = np.ascontiguousarray(received.imag.numpy().reshape(-1, 560), dtype=np.float32)
    pred = np.empty((len(yr), 320), dtype=np.uint8)
    margin = np.empty(pred.shape, dtype=np.float32)
    code = lib.run_slots(len(yr), yr.ctypes.data, yi.ctypes.data, pred.ctypes.data, margin.ctypes.data)
    if code:
        raise RuntimeError(f"Nonfinite native output at slot {code-1}")
    return pred.astype(bool), margin


def infer(model, y, variant):
    model.configure(variant)
    chunks = []
    with torch.inference_mode():
        for start in range(0, len(y), 64):
            p = model(y[start:start+64])
            if not torch.isfinite(p).all():
                raise ValueError("Non-finite PyTorch output")
            chunks.append((p[..., 1] > .5).reshape(-1, 320))
    return torch.cat(chunks).numpy()


def rounding_audit(lib, state):
    ties = np.arange(-2048, 2048, dtype=np.float32) + np.float32(.5)
    rng = np.random.default_rng(20261002)
    x = np.concatenate([ties, np.nextafter(ties, np.float32(-np.inf)),
                        np.nextafter(ties, np.float32(np.inf)),
                        np.arange(-2050, 2051, dtype=np.float32),
                        rng.uniform(-10000, 10000, 100000).astype(np.float32),
                        np.array([-np.finfo(np.float32).max, np.finfo(np.float32).max, -0., 0.], np.float32)])
    result = np.empty(x.shape, dtype=np.int32)
    lib.quantize12.argtypes = [ct.c_int, ct.c_void_p, ct.c_void_p]
    lib.quantize12(len(x), x.ctypes.data, result.ctypes.data)
    expected = torch.from_numpy(x).round().clamp(-2047, 2047).numpy().astype(np.int32)
    assert np.array_equal(result, expected)
    lib.get_scale12.argtypes = [ct.c_int]
    lib.get_scale12.restype = ct.c_float
    scales = {}
    for imaginary, part in enumerate(("qr", "qi")):
        value = float((state[f"eq.m_back.q_out.{part}.amax"] * 1.02).clamp_min(1e-8) / 2047)
        got = lib.get_scale12(imaginary)
        assert got == value
        scales[part] = value
    return {"samples": len(x), "mismatches": 0, "scales_exact_float32": scales,
            "nan_contract": "No NaN support claimed; all tested model outputs must be finite."}


def fixed_vectors(model, libs, output):
    y = torch.complex(torch.from_numpy(array("STIM_Y_RE").reshape(60, 7, 80)),
                      torch.from_numpy(array("STIM_Y_IM").reshape(60, 7, 80)))
    expected = {"rev_t": infer(model, y, "baseline_fft"), "a12": infer(model, y, "back_out_a12")}
    old = array("FFT_PRED", HERE / "stim_fft_preds.h", np.uint8).reshape(60, 320)
    assert np.array_equal(old, expected["rev_t"])
    rows, arrays = {}, {}
    for key, lib in libs.items():
        pred, margin = native(lib, y)
        indices = np.argwhere(pred != expected[key])
        rows[key] = {"decisions": 19200, "mismatches": len(indices),
                     "mismatch_indices": indices.tolist()}
        arrays.update({key + "_c": pred, key + "_torch": expected[key], key + "_margin": margin})
    header = "// Independent frozen-checkpoint FFT PyTorch A12 decisions.\n#pragma once\n#include <stdint.h>\n"
    flat = expected["a12"].astype(np.uint8).reshape(-1)
    header += "static const uint8_t A12_PRED[19200] = {\n"
    header += "\n".join("  " + ", ".join(map(str, flat[i:i+32])) + "," for i in range(0, len(flat), 32))
    (HERE / "stim_a12_preds.h").write_text(header + "\n};\n", encoding="ascii")
    rom = "// Independent FFT PyTorch local-A12 golden. Bit i = output i.\n"
    for slot in (0, 30):
        value = sum(int(v) << i for i, v in enumerate(expected["a12"][slot]))
        rom += f"localparam [319:0] PRED_S{slot} = 320'h{value:080x};\n"
    (HERE / "pred_roms_a12.vh").write_text(rom, encoding="ascii")
    np.savez_compressed(output / "fixed_vectors.npz", **arrays)
    save(output / "fixed_vectors.json", rows)
    print("fixed", rows, flush=True)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=HERE / "transfer_audit")
    parser.add_argument("--fixed-only", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Output exists; preserve prior evidence")
    args.output.mkdir(parents=True)
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    sources = [HERE / p for p in ("receiver_local_a12.cpp", "local_a12_quant.h", "native_bridge.cpp",
                "a12.dll", "rev_t.dll", "weights.h", "weights_pack_rev_f.h", "stim.h", "PROTOCOL.md", "validate_transfer.py")]
    sources += [BASE / "receiver_opt_rev_t.cpp", CKPT]
    sources += [ROOT / "dccn" / p for p in ("run_back_precision_followup.py", "run_precision_localization_pdpfix.py",
        "struct_quant_model.py", "quant.py", "quant_model.py", "struct_model.py", "config.py", "ofdm.py", "train.py", "paired_research_stats.py", "eval_paired_b1_longcp_pdpfix.py")]
    initial = {str(p): digest(p) for p in sources}
    for name in ("weights.h", "weights_pack_rev_f.h", "stim.h", "stim_fft_preds.h"):
        assert digest(HERE / name) == digest(BASE / name)
    manifest = {"status": "running", "sources": initial, "python": platform.python_version(),
                "torch": torch.__version__, "numpy": np.__version__, "compiler": "GCC 8.3.0 -O2 -ffp-contract=off",
                "checkpoint_scope": "BPSK long CP seed1 only", "conditions": [],
                "cohort": "Existing locked confirmation cohort 2026100202; transfer audit, not new independent samples"}
    save(args.output / "manifest.json", manifest)
    cfg = OfdmConfig(nbits=1, long_cp=True)
    state = torch.load(CKPT, map_location="cpu", weights_only=True)
    model = BackPrecisionReceiver(cfg, rank=0, eq_dense=True, slim=True, qc=QConfig()).eval()
    model.load_state_dict(state)
    libs = {name: library(name) for name in ("rev_t", "a12")}
    save(args.output / "quantizer.json", rounding_audit(libs["a12"], state))
    manifest["fixed"] = fixed_vectors(model, libs, args.output)
    rows = []
    if not args.fixed_only:
        for channel_name in ("EVA", "ETU"):
            for snr in (12, 16, 20, 24):
                started = time.perf_counter()
                name = f"b1_longcp_seed1_{channel_name}_{snr}dB"
                old = json.loads((PRIOR / (name + ".json")).read_text(encoding="utf-8"))
                seed = point_seed(2026100202, "b1_longcp", 1, channel_name, snr)
                assert seed == old["point_seed"]
                torch.manual_seed(seed)
                channel = ChannelTorch(cfg, torch.device("cpu"), model=channel_name)
                tx = OfdmTxTorch(cfg, torch.device("cpu"))
                with torch.inference_mode():
                    transmitted, labels, _ = tx.transmit(2000)
                    received, taps = apply_fading_awgn(channel, transmitted, torch.full((2000,), float(snr)))
                    noise = noise_residual(transmitted, received, taps, channel)
                hashes = {k: sha256_tensor(v) for k, v in (("transmitted", transmitted), ("labels", labels),
                    ("received", received), ("taps", taps), ("noise_residual", noise))}
                assert hashes == old["stimulus_sha256"], "Cohort identity mismatch"
                truth = labels.numpy().reshape(2000, 320).astype(bool)
                preds = {"torch_rev_t": infer(model, received, "baseline_fft"),
                         "torch_a12": infer(model, received, "back_out_a12")}
                data = {"labels": truth}
                for key, lib in libs.items():
                    preds["c_" + key], data[key + "_margins"] = native(lib, received)
                errors = {k: (v != truth).sum(1) for k, v in preds.items()}
                old_npz = np.load(PRIOR / (name + ".npz"))
                assert np.array_equal(errors["torch_rev_t"], old_npz["baseline_slot_errors"])
                assert np.array_equal(errors["torch_a12"], old_npz["back_out_a12__slot_errors"])
                row = {"condition": name, "slots": 2000, "bits": 640000, "stimulus_sha256": hashes,
                       "errors": {k: int(v.sum()) for k, v in errors.items()},
                       "c_vs_torch_flips": {k: int((preds["c_"+k] != preds["torch_"+k]).sum()) for k in libs},
                       "c_a12_vs_c_rev_t": summarize_paired_slots(errors["c_a12"], errors["c_rev_t"], 320,
                           bootstrap_reps=2000, bootstrap_seed=seed+884),
                       "torch_a12_vs_torch_rev_t": summarize_paired_slots(errors["torch_a12"], errors["torch_rev_t"], 320, bootstrap_reps=0)}
                for key, value in preds.items():
                    data[key + "_pred"] = value
                    data[key + "_errors"] = errors[key]
                np.savez_compressed(args.output / (name + ".npz"), **data)
                row["npz_sha256"] = digest(args.output / (name + ".npz"))
                row["elapsed_seconds"] = time.perf_counter() - started
                save(args.output / (name + ".json"), row)
                rows.append(row)
                manifest["conditions"].append(name)
                save(args.output / "manifest.json", manifest)
                print(name, row["errors"], row["c_vs_torch_flips"], f"{row['elapsed_seconds']:.1f}s", flush=True)
        total = {k: sum(r["errors"][k] for r in rows) for k in rows[0]["errors"]}
        software_gain = total["torch_rev_t"] - total["torch_a12"]
        c_gain = total["c_rev_t"] - total["c_a12"]
        summary = {"unique_slots": 16000, "unique_bits": 5120000, "total_errors": total,
                   "ber": {k: v/5120000 for k, v in total.items()},
                   "software_corrected_net": software_gain, "c_corrected_net": c_gain,
                   "retained_net_gain_ratio": c_gain/software_gain,
                   "c_relative_ber_reduction": c_gain/total["c_rev_t"],
                   "c_vs_torch_total_flips": {k: sum(r["c_vs_torch_flips"][k] for r in rows) for k in libs},
                   "conditions_c_a12_improves": sum(r["errors"]["c_a12"] < r["errors"]["c_rev_t"] for r in rows),
                   "limitations": "One fixed checkpoint; same prior confirmation inputs, not a fresh cohort or statistical equivalence claim. Native C mapping, not RTL on all these slots."}
        save(args.output / "summary.json", summary)
        print(json.dumps(summary), flush=True)
    assert all(torch.equal(value, state[name]) for name, value in model.state_dict().items())
    assert all(digest(p) == value for p, value in initial.items())
    manifest.update(status="complete", source_hashes_unchanged=True, state_unchanged=True)
    save(args.output / "manifest.json", manifest)


if __name__ == "__main__":
    main()
