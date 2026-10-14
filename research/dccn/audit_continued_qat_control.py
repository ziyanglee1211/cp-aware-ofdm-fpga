"""Independently audit continued-QAT artifacts without importing the runner.

Rebuilds bit/slot counts, six paired contrasts and all seed-grid means; verifies
original artifacts, full training state, activation ranges, training hash chains,
evaluation hashes and the output-only A12 contract. Does not run training, HLS,
Vivado or JTAG. Audit output is created only in a new directory.
"""

import argparse
import csv
import hashlib
import itertools
import json
import math
import platform
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from config import OfdmConfig
from ofdm import ChannelTorch, OfdmTxTorch
from quant import ActQ, QConfig
from struct_quant_model import StructuredQDCCN
from train import apply_fading_awgn

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
FORMAL_TAGS = ("b1_longcp", "b2_longcp", "b2_shortcp")
FORMAL_SEEDS = (1, 2, 3)
FORMAL_CHANNELS = ("EVA", "ETU")
FORMAL_SNRS = (12.0, 16.0, 20.0, 24.0)
ARMS = ("A0", "A1", "A2", "A1F", "A1A12")
TRAIN_ARMS = ("A1", "A1F")
CONTRASTS = ("A1_minus_A2", "A1_minus_A0", "A2_minus_A0",
             "A1F_minus_A0", "A1F_minus_A1", "A1A12_minus_A1")
A12_MODULES = {"eq.m_back.q_out.qr", "eq.m_back.q_out.qi"}
INPUT_BINDINGS = {}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def bind_input(path, expected=None):
    """Snapshot every consulted artifact; repeated reads must retain its hash."""
    path = Path(path).resolve(strict=True)
    digest = sha_file(path)
    if expected is not None:
        require(digest == expected, "File SHA-256 changed: " + str(path))
    key = str(path)
    require(key not in INPUT_BINDINGS or INPUT_BINDINGS[key] == digest,
            "Audit input changed between reads: " + key)
    INPUT_BINDINGS[key] = digest
    return digest


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True).encode("utf-8")


def tensor_hash(value):
    if isinstance(value, torch.Tensor):
        value = value.detach().contiguous().cpu().numpy()
    array = np.asarray(value)
    if not array.flags.c_contiguous:
        array = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()


def state_hash(state):
    digest = hashlib.sha256()
    for name in sorted(state):
        require(isinstance(state[name], torch.Tensor), "Non-tensor model state: " + name)
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(bytes.fromhex(tensor_hash(state[name])))
    return digest.hexdigest()


def load_json(path):
    bind_input(path)
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def save_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    allow_nan=False) + "\n", encoding="utf-8")


def resolve_path(value, base=ROOT):
    path = Path(value)
    return path.resolve(strict=True) if path.is_absolute() else (base / path).resolve(strict=True)


def checked_path(value, digest, base=ROOT):
    path = resolve_path(value, base)
    bind_input(path, digest)
    return path


def checked_local(value, base):
    path = resolve_path(value, base)
    require(base.resolve() in path.parents, "Artifact escapes evaluation directory: " + str(path))
    return path


def close(a, b, name, atol=1e-15):
    require(math.isfinite(float(a)) and math.isfinite(float(b)) and
            abs(float(a) - float(b)) <= atol, "Numerical mismatch: " + name)


def config(tag):
    require(tag in FORMAL_TAGS, "Unsupported config: " + tag)
    return OfdmConfig(nbits=int(tag[1]), long_cp=tag.endswith("longcp"))


def build_model(tag, state, arm, backend="fft"):
    model = StructuredQDCCN(config(tag), rank=0, eq_dense=True, slim=True,
                           qc=QConfig(wbits=8, abits=8))
    model.load_state_dict(state, strict=True)
    model.set_transform_backend(backend)
    if arm in ("A2", "A1A12"):
        model.eq.m_back.q_out.qr.bits = 12
        model.eq.m_back.q_out.qi.bits = 12
    return model.eval()


def activation_ranges(model):
    return {name: {"amax": float(module.amax.item()),
                   "calibrated": bool(module.calibrated.item()), "bits": int(module.bits)}
            for name, module in model.named_modules() if isinstance(module, ActQ)}


def check_ranges(recorded, actual, name):
    require(set(recorded) == set(actual), "Activation module set differs: " + name)
    require(len(actual) == 32, "Expected all 32 ActQ modules: " + name)
    for module, row in actual.items():
        require(recorded[module] == row, "Activation range differs: " + name + "/" + module)
        require(row["calibrated"] and math.isfinite(row["amax"]) and row["amax"] > 0,
                "Uncalibrated/nonfinite range: " + name + "/" + module)


def contract(arm, model, state):
    ranges = activation_ranges(model)
    expected_bits = {name: 12 if arm in ("A2", "A1A12") and name in A12_MODULES else 8
                     for name in ranges}
    require({name: row["bits"] for name, row in ranges.items()} == expected_bits,
            "Only A12 output boundary may have 12 bits: " + arm)
    require(all(module.qc.wbits == 8 for module in model.modules()
                if hasattr(module, "qc") and module.qc is not None), "Weights must remain W8")
    require(model.eq.transform_backend == model.basic.transform_backend,
            "Equalizer/basic transform backends differ")
    return {"activation_ranges": ranges, "activation_bits": expected_bits,
            "qconfig": asdict(model.qc),
            "wbits": 8, "backend": model.eq.transform_backend,
            "model_state_sha256": state_hash(state)}


def check_contract(recorded, expected, name):
    for field in ("activation_bits", "wbits", "backend", "model_state_sha256", "qconfig"):
        require(recorded[field] == expected[field], "Model contract differs: " + name + "/" + field)
    check_ranges(recorded["activation_ranges"], expected["activation_ranges"], name)


def check_sources(mapping):
    require(isinstance(mapping, dict) and mapping, "Missing source hash mapping")
    for name, digest in mapping.items():
        base = HERE if not Path(name).is_absolute() and "/" not in name and "\\" not in name else ROOT
        checked_path(name, digest, base)


def check_preserved(protocol):
    records = protocol.get("preserved_artifacts", [])
    if isinstance(records, dict):
        records = [{"path": key, "sha256": value} if isinstance(value, str)
                   else dict(value, path=value.get("path", key)) for key, value in records.items()]
    require(records, "Protocol must lock preserved_artifacts")
    checked = []
    for row in records:
        path = checked_path(row["path"], row["sha256"])
        checked.append({"path": str(path), "sha256": row["sha256"]})
    return checked


def chain_audit(folder, checkpoint):
    chain = hashlib.sha256(b"continued-qat-training-chain-v1").hexdigest()
    rows = []
    bind_input(folder / "stimuli.jsonl")
    with (folder / "stimuli.jsonl").open(encoding="utf-8") as stream:
        for index, line in enumerate(stream, 1):
            row = json.loads(line)
            require(row["step"] == index, "Training stimulus sequence not contiguous")
            hashes = row["stimulus_sha256"]
            require(set(hashes) == {"transmitted", "labels", "snr", "received", "taps"},
                    "Training stimulus hash fields differ")
            require(all(len(value) == 64 for value in hashes.values()), "Invalid stimulus SHA-256")
            if "taps_by_channel" in row:
                require(hashlib.sha256(canonical(row["taps_by_channel"])).hexdigest() == hashes["taps"],
                        "Training per-channel taps hash differs")
            chain = hashlib.sha256(bytes.fromhex(chain) + canonical(
                {"step": row["step"], "stimulus_sha256": hashes})).hexdigest()
            require(chain == row["chain_sha256"], "Training hash chain differs at step " + str(index))
            rows.append(row)
    require(len(rows) == checkpoint["step"] == checkpoint["expected_steps"],
            "Training step budget incomplete")
    require(chain == checkpoint["training_stimulus_chain_sha256"], "Final training chain differs")
    return rows, chain


def load_training(folder, tag, seed, arm, original, protocol_sha, protocol, smoke):
    manifest = load_json(folder / "manifest.json")
    require(manifest["status"] == "complete", "Incomplete training: " + str(folder))
    require(manifest["protocol_sha256"] == protocol_sha, "Training protocol SHA differs")
    require(manifest["source_hashes"] == protocol["source_hashes"], "Training source inventory differs")
    require(manifest["stage"] == "train" and not manifest["final_test_used"],
            "Training metadata reports final-test use or another stage")
    path = checked_path(manifest["final_checkpoint"]["path"],
                        manifest["final_checkpoint"]["sha256"], folder)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    require(payload["kind"] == "continued_qat_full_state", "Expected full training checkpoint")
    require((payload["tag"], payload["training_seed"], payload["arm"], payload["stage"]) ==
            (tag, seed, arm, "train"), "Training identity differs")
    require(payload["protocol_sha256"] == protocol_sha, "Checkpoint protocol SHA differs")
    require(sha_file(resolve_path(payload["protocol_path"])) == protocol_sha,
            "Checkpoint protocol path differs")
    require(payload["source_hashes"] == protocol["source_hashes"], "Checkpoint source inventory differs")
    require(not payload["final_test_used"], "Training checkpoint reports final-test use")
    if not smoke:
        require(payload["expected_steps"] == protocol["training"]["steps"] and
                payload["batch"] == protocol["training"]["batch"] and
                payload["training_slots"] == protocol["training"]["steps"] * protocol["training"]["batch"],
                "Training differs from locked update/example budget")
    for field in ("step", "training_slots", "training_stimulus_chain_sha256",
                  "initial_activation_ranges", "final_activation_ranges", "original_checkpoint"):
        require(manifest[field] == payload[field], "Manifest/checkpoint differs: " + field)
    require(payload["original_checkpoint"]["sha256"] == original["sha256"], "Wrong original checkpoint")
    require(resolve_path(payload["original_checkpoint"]["path"]) == original["path"],
            "Wrong original checkpoint path")
    require(payload["optimizer"]["state"] and payload["optimizer"]["param_groups"],
            "Missing optimizer state")
    require(set(payload["rng"]) >= {"torch_cpu", "torch_cuda", "python", "numpy"},
            "Missing full RNG state")
    require(isinstance(payload["rng"]["torch_cpu"], torch.Tensor) and
            payload["rng"]["torch_cpu"].dtype == torch.uint8 and
            payload["rng"]["torch_cpu"].numel() > 0 and
            isinstance(payload["rng"]["python"], tuple) and
            payload["rng"]["numpy"]["kind"] == "MT19937" and
            payload["rng"]["numpy"]["keys"].shape == (624,), "Malformed complete RNG state")
    require(payload["scheduler"]["last_epoch"] == payload["step"], "Scheduler step budget differs")
    require(payload["scheduler"]["step_size"] == protocol["training"]["step_size"] and
            payload["scheduler"]["gamma"] == protocol["training"]["gamma"], "Scheduler policy differs")
    require(payload["observer_mode"] == protocol["training"]["observer_modes"][arm] and
            payload["selection"] == "final", "Training range/checkpoint selection policy differs")
    for group in payload["optimizer"]["param_groups"]:
        close(group["weight_decay"], protocol["training"]["weight_decay"], "Optimizer weight decay")
        require(group["initial_lr"] == protocol["training"]["lr"], "Optimizer initial LR differs")
        require(tuple(group["betas"]) == (0.9, 0.999) and group["eps"] == 1e-8 and
                not group["amsgrad"], "Fresh Adam default policy differs")
        close(group["lr"], protocol["training"]["lr"] *
              protocol["training"]["gamma"] ** (payload["step"] // protocol["training"]["step_size"]),
              "Optimizer final LR")
    for row in payload["optimizer"]["state"].values():
        require(set(row) == {"step", "exp_avg", "exp_avg_sq"}, "Adam full-state fields differ")
        require(int(row["step"]) == payload["step"], "Optimizer step budget differs")
        require(all(bool(torch.isfinite(value).all()) for value in row.values()
                    if isinstance(value, torch.Tensor)), "Nonfinite optimizer tensor")
    check_sources(payload["source_hashes"])
    initial_model = build_model(tag, original["state"], "A0", protocol["training"]["backend"])
    final_model = build_model(tag, payload["model"], arm, protocol["training"]["backend"])
    parameters = set(dict(final_model.named_parameters()))
    require(set(original["state"]) == set(payload["model"]), "Continued model state fields differ")
    for name, value in payload["model"].items():
        require(value.shape == original["state"][name].shape and value.dtype == original["state"][name].dtype,
                "Continued tensor shape/dtype differs: " + name)
        require(bool(torch.isfinite(value).all()), "Nonfinite continued tensor: " + name)
        if name not in parameters and not name.endswith((".amax", ".calibrated")):
            require(torch.equal(value, original["state"][name]), "Fixed backbone/buffer changed: " + name)
    initial = activation_ranges(initial_model)
    final = activation_ranges(final_model)
    check_ranges(payload["initial_activation_ranges"], initial, tag + "/" + arm + "/initial")
    check_ranges(payload["final_activation_ranges"], final, tag + "/" + arm + "/final")
    expected = contract(arm, final_model, payload["model"])
    check_contract(payload["model_contract"], expected, tag + "/" + arm)
    if arm == "A1F":
        require(initial == final, "Frozen-range diagnostic changed activation maxima")
    else:
        require(all(final[name]["amax"] >= initial[name]["amax"] for name in initial),
                "Primary continued range unexpectedly shrank")
    chain_rows, chain = chain_audit(folder, payload)
    locked_seed = protocol["rng"]["training_seeds"][f"{tag}_seed{seed}"]
    require(all(row["training_rng_seed"] == locked_seed for row in chain_rows),
            "Training random namespace differs from protocol")
    batch_sizes = {int(row.get("batch", payload["training_slots"] // payload["step"]))
                   for row in chain_rows}
    require(len(batch_sizes) == 1 and payload["training_slots"] == payload["step"] * next(iter(batch_sizes)),
            "Training example budget differs")
    bind_input(folder / "training.jsonl")
    metrics = [json.loads(line) for line in (folder / "training.jsonl").read_text(encoding="utf-8").splitlines()
               if line.strip()]
    require(metrics and all(1 <= int(row["step"]) <= payload["step"] for row in metrics),
            "Training metrics log incomplete")
    expected_metric_steps = sorted(set(range(protocol["training"]["log_every"], payload["step"] + 1,
                                            protocol["training"]["log_every"])) | {payload["step"]})
    require([row["step"] for row in metrics] == expected_metric_steps, "Training metrics steps missing/repeated")
    previous_ranges = initial
    for row in metrics:
        require(row["training_stimulus_chain_sha256"] == chain_rows[row["step"] - 1]["chain_sha256"],
                "Training metric hash-chain binding differs")
        require(row["training_slots"] == row["step"] * next(iter(batch_sizes)), "Training metric example count differs")
        require(math.isfinite(row["loss"]) and math.isfinite(row["train_batch_ber"]),
                "Nonfinite training metric")
        close(row["lr"], protocol["training"]["lr"] *
              protocol["training"]["gamma"] ** (row["step"] // protocol["training"]["step_size"]),
              "Training metric learning rate")
        ranges = row["activation_ranges"]
        check_ranges(ranges, ranges, tag + "/" + arm + "/metric_ranges")
        require(set(ranges) == set(initial) and all(ranges[name]["bits"] == 8 for name in initial),
                "Training metric quantizer contract differs")
        require(all(ranges[name]["amax"] >= previous_ranges[name]["amax"] for name in initial),
                "Running activation range decreased in metric history")
        if arm == "A1F":
            require(ranges == initial, "Frozen-range diagnostic changed in metric history")
        previous_ranges = ranges
    require(previous_ranges == final, "Final metric/checkpoint activation ranges differ")
    resume_step = 0
    if "resume_from" in manifest:
        source = manifest["resume_from"]
        resume_path = checked_path(source["path"], source["sha256"], folder)
        resume_payload = torch.load(resume_path, map_location="cpu", weights_only=True)
        require(resume_payload["step"] == source["step"] and
                resume_payload["protocol_sha256"] == protocol_sha and
                (resume_payload["tag"], resume_payload["training_seed"], resume_payload["arm"]) ==
                (tag, seed, arm), "Resume checkpoint provenance differs")
        resume_step = source["step"]
        require(0 < resume_step < payload["step"] and
                resume_payload["training_stimulus_chain_sha256"] ==
                chain_rows[resume_step - 1]["chain_sha256"],
                "Resume checkpoint differs from retained training stimulus prefix")
    checkpoint_steps = [step for step in range(protocol["training"]["checkpoint_every"], payload["step"],
                                               protocol["training"]["checkpoint_every"]) if step > resume_step]
    require([row["step"] for row in manifest["checkpoints"]] == checkpoint_steps,
            "Intermediate checkpoint inventory missing/repeated")
    for row in manifest["checkpoints"]:
        checked_path(row["path"], row["sha256"], folder)
    eval_model = build_model(tag, payload["model"], arm)
    expected_eval = contract(arm, eval_model, payload["model"])
    return {"state": payload["model"], "contract": expected_eval, "checkpoint": path,
            "sha256": sha_file(path), "chain_rows": chain_rows, "chain": chain,
            "steps": payload["step"], "slots": payload["training_slots"],
            "amax_changed": [name for name in initial if initial[name] != final[name]],
            "weights_changed": [name for name in original["state"]
                                if not name.endswith((".amax", ".calibrated")) and
                                not torch.equal(original["state"][name], payload["model"][name])]}


def noise_residual(transmitted, received, taps, channel):
    n, f, s = transmitted.shape
    flat = transmitted.reshape(n, -1)
    if channel.L == 1:
        faded = flat * taps
    else:
        length = 1 << (flat.shape[1] + channel.L - 2).bit_length()
        faded = torch.fft.ifft(torch.fft.fft(flat, n=length, dim=1) *
                                 torch.fft.fft(taps, n=length, dim=1), dim=1)[:, :flat.shape[1]]
    return received.reshape(n, -1) - faded


def regenerate_traffic(point):
    torch.manual_seed(point["point_seed"])
    cfg = config(point["tag"])
    device = torch.device("cpu")
    channel = ChannelTorch(cfg, device, model=point["channel"])
    tx = OfdmTxTorch(cfg, device)
    with torch.inference_mode():
        transmitted, labels, _ = tx.transmit(point["slots"])
        received, taps = apply_fading_awgn(channel, transmitted,
            torch.full((point["slots"],), float(point["snr_db"])))
        noise = noise_residual(transmitted, received, taps, channel)
    hashes = {name: tensor_hash(value) for name, value in (
        ("transmitted", transmitted), ("labels", labels), ("received", received),
        ("taps", taps), ("noise_residual", noise))}
    require(hashes == point["stimulus_sha256"], "Regenerated traffic SHA differs: " + str(point["point_seed"]))
    return labels.bool().reshape(point["slots"], -1).numpy()


def old_received_hashes(protocol):
    result = set()
    required = {path.resolve() for path in (
        HERE / "experiments/precision_localization_20261001/screen_2000",
        HERE / "experiments/back_precision_followup_20261002/independent_2000",
        HERE / "experiments/back_precision_followup_20261002/confirmation_2000")}
    folders = {resolve_path(value) for value in protocol["historical_cohort_dirs"]}
    require(required.issubset(folders), "Protocol omits an earlier screening/confirmation cohort")
    for folder in folders:
        paths = list(folder.glob("*_seed*dB.json"))
        require(paths, "Empty earlier cohort: " + str(folder))
        for path in paths:
            result.add(load_json(path)["stimulus_sha256"]["received"])
    require(result, "Earlier cohort hash inventory missing")
    return result


def audit_evaluation(folder, manifest, states, settings, verify_traffic, protocol):
    expected_keys = set(itertools.product(settings["tags"], settings["seeds"],
                                         settings["channels"], settings["snrs"]))
    names = manifest["conditions"]
    require(len(names) == len(set(names)) == len(expected_keys), "Condition count/duplicates differ")
    seen, stimuli, namespaces = set(), set(), set()
    old_hashes = old_received_hashes(protocol)
    training_hashes = {row["stimulus_sha256"]["received"]
                       for state in states.values() for arm in TRAIN_ARMS
                       for row in state[arm]["chain_rows"]}
    raw = []
    total = {arm: 0 for arm in ARMS}
    bits_total, slots_total = 0, 0
    for filename in names:
        row_path = checked_local(filename, folder)
        point = load_json(row_path)
        require(point["protocol_sha256"] == sha_file(Path(point["protocol_path"])) and
                point["protocol_sha256"] == manifest["protocol_sha256"],
                "Condition protocol binding differs")
        key = (point["tag"], point["training_seed"], point["channel"], float(point["snr_db"]))
        require(key in expected_keys and key not in seen, "Missing/unexpected/repeated condition: " + str(key))
        seen.add(key)
        require(point["slots"] == settings["slots"], "Slots per condition differ")
        require(point["bits_per_slot"] == config(point["tag"]).D_slot * int(point["tag"][1]),
                "Bits per slot denominator differs")
        require(point["pdp_mode"] == "3gpp_db_to_linear_power", "Wrong PDP generator mode")
        received_hash = point["stimulus_sha256"]["received"]
        require(received_hash not in old_hashes | training_hashes | stimuli,
                "Evaluation received tensor reuses an old or current cohort")
        require(point["point_seed"] not in namespaces, "Evaluation random namespace repeats")
        namespace_key = f"{point['tag']}_seed{point['training_seed']}_{point['channel']}_{point['snr_db']:g}dB"
        require(row_path.name == namespace_key + ".json", "Condition filename differs from identity")
        require(point["point_seed"] == protocol["rng"]["final_test_seeds"][namespace_key],
                "Evaluation namespace differs from locked final-test seed")
        stimuli.add(received_hash)
        namespaces.add(point["point_seed"])
        require(set(point["arms"]) == set(ARMS) and set(point["contrasts"]) == set(CONTRASTS),
                "Every predeclared arm/contrast must be retained")
        expected_states = states[(point["tag"], point["training_seed"])]
        for arm in ARMS:
            check_contract(point["arm_contracts"][arm], expected_states[arm]["contract"],
                           filename + "/" + arm)
            require(point["checkpoint_sha256"][arm] == expected_states[arm]["sha256"],
                    "Evaluation checkpoint differs: " + filename + "/" + arm)
        npz_path = checked_local(point["slot_data"], folder)
        bind_input(npz_path, point["slot_data_sha256"])
        labels = regenerate_traffic(point) if verify_traffic else None
        with np.load(npz_path, allow_pickle=False) as data:
            truth = data["labels"]
            require(truth.dtype == np.bool_ and truth.shape == (point["slots"], point["bits_per_slot"]),
                    "Raw labels shape/dtype differs")
            if labels is not None:
                require(np.array_equal(labels, truth), "Stored truth differs from regenerated labels")
            errors, predictions = {}, {}
            for arm in ARMS:
                require(point["arms"][arm]["slots"] == point["slots"] and
                        point["arms"][arm]["bits_per_slot"] == point["bits_per_slot"],
                        "Arm sample-unit metadata differs: " + filename + "/" + arm)
                pred = data[arm + "__pred"]
                require(pred.dtype == np.bool_ and pred.shape == truth.shape, "Raw predictions malformed: " + arm)
                predictions[arm] = pred
                errors[arm] = (pred != truth).sum(axis=1, dtype=np.int64)
                require(np.array_equal(errors[arm], data[arm + "__slot_errors"]),
                        "Slot errors differ from raw decisions: " + filename + "/" + arm)
                count = int(errors[arm].sum())
                require(count == point["arms"][arm]["error_bits"], "Arm error bits differ")
                close(count / truth.size, point["arms"][arm]["ber"], filename + "/" + arm + "/BER")
                total[arm] += count
            deltas_for_bootstrap = []
            for contrast in CONTRASTS:
                candidate, reference = contrast.split("_minus_")
                cand_wrong, ref_wrong = predictions[candidate] != truth, predictions[reference] != truth
                corrected = (ref_wrong & ~cand_wrong).sum(axis=1, dtype=np.int64)
                harmed = (~ref_wrong & cand_wrong).sum(axis=1, dtype=np.int64)
                flips = (predictions[candidate] != predictions[reference]).sum(axis=1, dtype=np.int64)
                for suffix, values in (("corrected", corrected), ("harmed", harmed), ("flips", flips)):
                    require(np.array_equal(values, data[contrast + "__" + suffix]),
                            "Paired raw count differs: " + filename + "/" + contrast + "/" + suffix)
                require(np.array_equal(flips, corrected + harmed) and
                        np.array_equal(errors[candidate] - errors[reference], harmed - corrected),
                        "Paired accounting identities fail")
                delta = (errors[candidate] - errors[reference]).astype(np.float64) / point["bits_per_slot"]
                recorded = point["contrasts"][contrast]
                require(recorded["bootstrap_reps"] == protocol["evaluation"]["bootstrap"] and
                        recorded["bootstrap_seed"] == point["point_seed"] +
                        protocol["evaluation"]["bootstrap_seed_offset"], "Conditional bootstrap settings differ")
                require(recorded["candidate_error_bits"] == int(errors[candidate].sum()) and
                        recorded["reference_error_bits"] == int(errors[reference].sum()), "Contrast error bits differ")
                close(delta.mean(), recorded["candidate_minus_reference_ber"], filename + "/" + contrast)
                close(delta.std(ddof=1), recorded["slot_delta_sample_sd"], filename + "/" + contrast + "/SD")
                for field, count in (("changed_decisions", int(flips.sum())),
                                     ("corrected_decisions", int(corrected.sum())),
                                     ("harmed_decisions", int(harmed.sum()))):
                    if field in recorded:
                        require(recorded[field] == count, "Recorded decision count differs: " + field)
                raw.append({"tag": point["tag"], "training_seed": point["training_seed"],
                            "channel": point["channel"], "snr_db": point["snr_db"],
                            "contrast": contrast, "delta_ber": float(delta.mean()),
                            "candidate_errors": int(errors[candidate].sum()),
                            "reference_errors": int(errors[reference].sum()), "bits": truth.size,
                            "corrected": int(corrected.sum()), "harmed": int(harmed.sum()),
                            "flips": int(flips.sum()), "npz_sha256": point["slot_data_sha256"]})
                deltas_for_bootstrap.append(delta)
            # Shared draws are reconstructed without the runner or its statistics helper.
            reps = protocol["evaluation"]["bootstrap"]
            delta_matrix = np.stack(deltas_for_bootstrap, axis=1)
            rng = np.random.default_rng(point["point_seed"] + protocol["evaluation"]["bootstrap_seed_offset"])
            means = np.empty((reps, len(CONTRASTS)), dtype=np.float64)
            for start in range(0, reps, 128):
                stop = min(start + 128, reps)
                indices = rng.integers(0, point["slots"], size=(stop - start, point["slots"]))
                means[start:stop] = delta_matrix[indices].mean(axis=1)
            intervals = np.quantile(means, [0.025, 0.975], axis=0)
            for index, contrast in enumerate(CONTRASTS):
                reported = point["contrasts"][contrast]["conditional_slot_bootstrap_95pct_ci"]
                close(intervals[0, index], reported[0], filename + "/" + contrast + "/bootstrap_low")
                close(intervals[1, index], reported[1], filename + "/" + contrast + "/bootstrap_high")
            bits_total += truth.size
            slots_total += len(truth)
    require(seen == expected_keys, "Condition Cartesian grid incomplete")
    require(manifest["unique_slots"] == slots_total and manifest["unique_bits"] == bits_total,
            "Manifest sample totals differ")
    return raw, total, slots_total, bits_total, len(old_hashes)


def grid_summary(raw, settings):
    result = []
    for tag, contrast in itertools.product(settings["tags"], CONTRASTS):
        values, per_seed = [], []
        for seed in settings["seeds"]:
            rows = [row for row in raw if row["tag"] == tag and row["training_seed"] == seed and
                    row["contrast"] == contrast]
            require(len(rows) == len(settings["channels"]) * len(settings["snrs"]), "Seed grid incomplete")
            value = float(np.mean([row["delta_ber"] for row in rows]))
            reference_ber = sum(row["reference_errors"] for row in rows) / sum(row["bits"] for row in rows)
            candidate_ber = sum(row["candidate_errors"] for row in rows) / sum(row["bits"] for row in rows)
            values.append(value)
            per_seed.append({"training_seed": seed, "delta_ber": value,
                             "candidate_grid_ber": candidate_ber, "reference_grid_ber": reference_ber,
                             "relative_change_percent": 100 * value / reference_ber if reference_ber else None})
        result.append({"tag": tag, "contrast": contrast, "raw_seed_deltas": values,
                       "mean_delta": float(np.mean(values)),
                       "sample_sd": float(np.std(values, ddof=1)) if len(values) > 1 else None,
                       "all_raw_seeds": per_seed})
    return result


def audit_zero(protocol_path, protocol, digest, states):
    path = protocol_path.parent / "zero/manifest.json"
    zero = load_json(path)
    require(zero["status"] == "complete" and zero["stage"] == "zero" and
            zero["protocol_sha256"] == digest and not zero["final_test_used"],
            "Zero-update development gate is incomplete or used final traffic")
    expected_keys = set(states)
    seen = set()
    for row in zero["conditions"]:
        key = (row["tag"], row["training_seed"])
        require(key in expected_keys and key not in seen, "Zero-update task set differs")
        seen.add(key)
        require(row["development_rng_seed"] == protocol["rng"]["development_seeds"][
                    f"{row['tag']}_seed{row['training_seed']}"], "Zero-update RNG namespace differs")
        require(set(row["arms"]) == set(ARMS), "Zero-update gate omits an inference arm")
        original = states[key]["A0"]["state"]
        for arm in ARMS:
            require(row["arms"][arm]["max_absolute_difference"] == 0,
                    "Zero-update arm did not exactly match its original reference")
            model = build_model(row["tag"], original, arm)
            check_contract(row["arms"][arm]["model_contract"], contract(arm, model, original),
                           "zero/" + str(key) + "/" + arm)
    require(seen == expected_keys, "Not all nine zero-update tasks were retained")
    return {"status": "pass", "tasks": len(seen), "arms_per_task": len(ARMS),
            "manifest_sha256": sha_file(path), "final_test_used": False}


def read_settings(protocol):
    values = protocol["evaluation"]
    return {"tags": list(protocol["tags"]), "seeds": list(map(int, protocol["seeds"])),
            "channels": list(values["channels"]), "snrs": list(map(float, values["snrs"])),
            "slots": int(values["slots"])}


def write_csv(path, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True, help="Evaluation output directory")
    parser.add_argument("--train-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-smoke", action="store_true", help="Clearly label subset checks as nonpublication")
    parser.add_argument("--skip-traffic-regeneration", action="store_true",
                        help="Only for smoke; formal audit must regenerate all five input hashes")
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    INPUT_BINDINGS.clear()
    bind_input(Path(__file__))
    require(args.threads > 0, "Threads must be positive")
    require(not args.skip_traffic_regeneration or args.allow_smoke, "Formal audits must regenerate traffic")
    output = args.output.resolve()
    require(not output.exists(), "Preserve existing audit; use a new output directory")
    folder, train_root = args.run.resolve(strict=True), args.train_root.resolve(strict=True)
    protocol_path = args.protocol.resolve(strict=True)
    protocol, protocol_sha = load_json(protocol_path), sha_file(protocol_path)
    require(protocol["status"] == "locked_before_execution", "Protocol was not locked before execution")
    lock = load_json(protocol_path.parent / "protocol_lock.json")
    require(lock["sha256"] == protocol_sha and lock["locked_before_zero_pilot_train_final"],
            "Protocol lock differs")
    check_sources(protocol["source_hashes"])
    preserved = check_preserved(protocol)
    manifest = load_json(folder / "manifest.json")
    require(manifest["status"] == "complete", "Require a completed evaluation")
    require(manifest["protocol_sha256"] == protocol_sha, "Evaluation protocol SHA differs")
    smoke = bool(manifest.get("smoke_not_for_publication", manifest.get("smoke", False)))
    require(not smoke or args.allow_smoke, "Smoke evaluation cannot pass formal audit")
    require(manifest["stage"] == "evaluate" and manifest["final_test_used"], "Not a final-test evaluation")
    require(manifest["source_hashes"] == protocol["source_hashes"], "Evaluation source inventory differs")
    settings = read_settings(protocol)
    if "args" in manifest:
        require(all(manifest["args"][field] == value for field, value in settings.items()),
                "Evaluation manifest settings differ from protocol")
        require(manifest["args"]["batch"] == protocol["evaluation"]["batch"] and
                manifest["args"]["bootstrap"] == protocol["evaluation"]["bootstrap"] and
                resolve_path(manifest["args"]["training_root"]) == train_root,
                "Evaluation batch/bootstrap/training-root binding differs")
    if not args.allow_smoke:
        require(settings["tags"] == protocol["tags"] and settings["seeds"] == protocol["seeds"],
                "Evaluation configs/seeds differ from protocol")
        require(all(settings[field] == protocol["evaluation"][field]
                    for field in ("channels", "snrs", "slots")), "Evaluation grid differs from protocol")
    for field in ("tags", "seeds", "channels", "snrs"):
        require(len(settings[field]) == len(set(settings[field])), "Repeated setting: " + field)
    if not args.allow_smoke:
        require(tuple(settings["tags"]) == FORMAL_TAGS and tuple(settings["seeds"]) == FORMAL_SEEDS and
                tuple(settings["channels"]) == FORMAL_CHANNELS and tuple(settings["snrs"]) == FORMAL_SNRS and
                settings["slots"] == 2000, "Formal 72-condition / 9-run specification differs")
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    check_sources(manifest["source_hashes"])
    task_names = {f"{tag}_seed{seed}" for tag, seed in
                  itertools.product(settings["tags"], settings["seeds"])}
    require(set(manifest["original_checkpoints"]) == task_names,
            "Evaluation original-checkpoint task inventory differs")
    require(set(manifest["training_checkpoints"]) ==
            {name + "_" + arm for name in task_names for arm in TRAIN_ARMS},
            "Evaluation continued-training task inventory differs")
    states, training_rows = {}, []
    for tag, seed in itertools.product(settings["tags"], settings["seeds"]):
        name = f"{tag}_seed{seed}"
        path = HERE / "ckpt" / f"sq_{tag}_s_r0_slim_w8a8_pdpfix_seed{seed}.pt"
        sha = sha_file(path)
        recorded = manifest.get("original_checkpoints", manifest.get("checkpoints", {}))[name]
        require(recorded["sha256"] == sha and resolve_path(recorded["path"]) == path.resolve(),
                "Original checkpoint identity differs: " + name)
        require(any(row["path"] == str(path.resolve()) and row["sha256"] == sha for row in preserved),
                "Original checkpoint not locked before training: " + name)
        original = {"path": path.resolve(), "sha256": sha,
                    "state": torch.load(path, map_location="cpu", weights_only=True)}
        run_states = {}
        for arm in ("A0", "A2"):
            model = build_model(tag, original["state"], arm)
            run_states[arm] = {"state": original["state"], "sha256": sha,
                               "contract": contract(arm, model, original["state"])}
        require(run_states["A0"]["contract"]["model_state_sha256"] ==
                run_states["A2"]["contract"]["model_state_sha256"], "A2 original state differs")
        for arm in TRAIN_ARMS:
            trained = load_training(train_root / name / arm, tag, seed, arm, original,
                                    protocol_sha, protocol, args.allow_smoke)
            run_states[arm] = trained
            binding = manifest["training_checkpoints"][name + "_" + arm]
            require(resolve_path(binding["path"]) == trained["checkpoint"] and
                    binding["sha256"] == trained["sha256"], "Evaluation training binding differs")
            require(binding["training_manifest_sha256"] == sha_file(train_root / name / arm / "manifest.json"),
                    "Evaluation final-training manifest binding differs")
            training_rows.append({"tag": tag, "training_seed": seed, "arm": arm,
                                  "checkpoint_path": str(trained["checkpoint"]),
                                  "checkpoint_sha256": trained["sha256"], "steps": trained["steps"],
                                  "training_slots": trained["slots"], "training_chain_sha256": trained["chain"],
                                  "amax_changed": trained["amax_changed"],
                                  "weights_changed": trained["weights_changed"]})
        require(run_states["A1"]["chain"] == run_states["A1F"]["chain"] and
                run_states["A1"]["chain_rows"] == run_states["A1F"]["chain_rows"],
                "Primary/frozen-range training inputs differ")
        require(run_states["A1"]["steps"] == run_states["A1F"]["steps"] and
                run_states["A1"]["slots"] == run_states["A1F"]["slots"], "Training budgets differ")
        state = run_states["A1"]["state"]
        model = build_model(tag, state, "A1A12")
        run_states["A1A12"] = {"state": state, "sha256": run_states["A1"]["sha256"],
                                "contract": contract("A1A12", model, state)}
        states[(tag, seed)] = run_states
    raw, totals, slots, bits, old_conditions = audit_evaluation(
        folder, manifest, states, settings, not args.skip_traffic_regeneration, protocol)
    grid = grid_summary(raw, settings)
    zero_audit = audit_zero(protocol_path, protocol, protocol_sha, states)
    for state in states.values():
        for arm in TRAIN_ARMS:
            require(sha_file(state[arm]["checkpoint"]) == state[arm]["sha256"],
                    "Continued checkpoint changed during audit")
    after = check_preserved(protocol)
    require(preserved == after, "Original artifact preservation changed during audit")
    for path, digest in list(INPUT_BINDINGS.items()):
        require(sha_file(path) == digest, "Audit input changed during audit: " + path)
    report = {"status": "pass", "smoke_not_for_publication": args.allow_smoke or smoke,
              "conditions": len(manifest["conditions"]), "primary_training_runs": len(states),
              "frozen_range_diagnostic_runs": len(states), "unique_slots": slots, "unique_bits": bits,
              "arms": list(ARMS), "contrasts": list(CONTRASTS), "error_bits_recomputed": totals,
              "all_raw_seed_grid_contrasts": grid, "training_runs": training_rows,
              "zero_update_development_gate": zero_audit,
              "earlier_cohort_received_hashes_checked": old_conditions,
              "all_input_hashes_regenerated": not args.skip_traffic_regeneration,
              "input_hash_regeneration_scope": "All final evaluation conditions; training inputs checked by recorded hash-chain reconstruction, not replayed",
              "training_stimulus_chains_rebuilt": True,
              "model_inference_replayed": False,
              "all_raw_predictions_and_paired_counts_rebuilt": True,
              "all_conditional_slot_bootstrap_intervals_rebuilt": True,
              "A2_only_output_quantizer_bits_changed": True,
              "A1A12_only_output_quantizer_bits_changed": True,
              "original_artifacts_preserved": preserved,
              "protocol_path": str(protocol_path), "protocol_sha256": protocol_sha,
              "evaluation_manifest_sha256": sha_file(folder / "manifest.json"),
              "input_bindings": [{"path": path, "sha256": digest}
                                 for path, digest in sorted(INPUT_BINDINGS.items())],
              "auditor_sha256": sha_file(Path(__file__)),
              "environment": {"python": platform.python_version(), "python_executable": sys.executable,
                              "torch": torch.__version__, "numpy": np.__version__, "threads": args.threads},
              "scope": "Software-only independent artifact/count audit; no C/RTL, implementation or board validation."}
    output.mkdir(parents=True)
    write_csv(output / "condition_contrasts.csv", raw)
    save_json(output / "grid_raw_seed_summary.json", grid)
    save_json(output / "audit.json", report)
    print(json.dumps({key: report[key] for key in ("status", "smoke_not_for_publication", "conditions",
                     "primary_training_runs", "frozen_range_diagnostic_runs", "unique_slots", "unique_bits")},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
