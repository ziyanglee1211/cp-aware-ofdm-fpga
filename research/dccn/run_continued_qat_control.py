"""Protocol-bound continuation controls; never overwrite historical evidence.

zero and pilot consume only development RNG seeds. train starts from the
selected corrected-PDP QAT weights with a new optimizer, or resumes a complete
checkpoint produced here. evaluate requires an explicit final-test flag and
all fixed-budget training tasks. The protocol file and its digest are required
for every command. No test performance selects training steps or checkpoints.
"""
import argparse
import copy
import ctypes
import hashlib
import itertools
import json
import os
import platform
import random
import sys
import time
import uuid
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from config import OfdmConfig
from ofdm import ChannelTorch, OfdmTxTorch
from paired_research_stats import summarize_paired_slots
from quant import ActQ, QConfig
from receiver import ber, ce_loss
from struct_quant_model import StructuredQDCCN
from train import apply_fading_awgn, sample_mixed_snr

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
TAGS = ("b1_longcp", "b2_longcp", "b2_shortcp")
SEEDS = (1, 2, 3)
CHANNELS = ("flat", "EPA", "EVA", "ETU")
BOUNDARY = ("eq.m_back.q_out.qr", "eq.m_back.q_out.qi")
CHAIN_INITIAL = hashlib.sha256(b"continued-qat-training-chain-v1").hexdigest()
REQUIRED_SOURCES = tuple("dccn/" + name + ".py" for name in (
    "run_continued_qat_control", "config", "ofdm", "train", "quant",
    "quant_model", "struct_model", "struct_quant_model", "receiver",
    "paired_research_stats", "eval_struct_qat")) + tuple(
        "code/dl_ofdm-master/dev/py/3gpp/AM_" + name + ".csv"
        for name in ("EPA", "EVA", "ETU"))


def canonical(data):
    return json.dumps(data, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True).encode("utf-8")


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tensor_hash(value):
    array = value.detach().contiguous().cpu().numpy()
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()


def state_hash(state):
    digest = hashlib.sha256()
    for name in sorted(state):
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(bytes.fromhex(tensor_hash(state[name])))
    return digest.hexdigest()


def advance_chain(previous, step, hashes):
    record = {"step": step, "stimulus_sha256": hashes}
    return hashlib.sha256(bytes.fromhex(previous) + canonical(record)).hexdigest()


def task_key(tag, seed):
    return f"{tag}_seed{seed}"


def condition_key(tag, seed, channel, snr):
    return f"{task_key(tag, seed)}_{channel}_{snr:g}dB"


def environment():
    return {"python": platform.python_version(), "torch": str(torch.__version__),
            "numpy": np.__version__, "device": "cpu",
            "threads": torch.get_num_threads(), "python_executable": sys.executable,
            "platform": platform.platform(),
            "torch_interop_threads": torch.get_num_interop_threads()}


def verify_bindings(protocol):
    for name, expected in protocol["source_hashes"].items():
        path = Path(name)
        if not path.is_absolute():
            path = REPO / path
        if file_hash(path) != expected:
            raise ValueError("Source/input hash changed: " + str(path))
    for key, row in protocol["checkpoints"].items():
        if file_hash(row["path"]) != row["sha256"]:
            raise ValueError("Original checkpoint hash changed: " + key)


def validate_protocol(protocol):
    if protocol.get("schema_version") != 1 or protocol.get("status") != "locked_before_execution":
        raise ValueError("A locked version-1 protocol is required")
    if tuple(protocol["tags"]) != TAGS or tuple(protocol["seeds"]) != SEEDS:
        raise ValueError("The complete three-configuration, three-seed cohort is required")
    expected_tasks = {task_key(t, s) for t, s in itertools.product(TAGS, SEEDS)}
    if set(protocol["checkpoints"]) != expected_tasks:
        raise ValueError("Expected all nine original checkpoints")
    for tag, seed in itertools.product(TAGS, SEEDS):
        path = HERE / "ckpt" / f"sq_{tag}_s_r0_slim_w8a8_pdpfix_seed{seed}.pt"
        if Path(protocol["checkpoints"][task_key(tag, seed)]["path"]).resolve() != path.resolve():
            raise ValueError("Unexpected original checkpoint path")
    if not set(REQUIRED_SOURCES).issubset(protocol["source_hashes"]):
        raise ValueError("Protocol must bind all receiver sources and channel CSV inputs")
    tr = protocol["training"]
    fixed = {"steps": 3000, "batch": 512, "lr": 1e-4, "weight_decay": 1e-5,
             "step_size": 500, "gamma": .95, "backend": "matrix", "selection": "final"}
    if any(tr.get(key) != value for key, value in fixed.items()):
        raise ValueError("Training contract differs from the fixed continuation protocol")
    if tuple(tr["channels"]) != CHANNELS:
        raise ValueError("Original mixed-channel training law must be retained")
    if "A1" not in tr["arms"] or not set(tr["arms"]).issubset({"A1", "A1F"}):
        raise ValueError("Only A1 and predeclared A1F train")
    if len(tr["arms"]) != len(set(tr["arms"])):
        raise ValueError("Duplicate training arms")
    for arm in tr["arms"]:
        if tr["observer_modes"][arm] != {"A1": "running_max", "A1F": "frozen"}[arm]:
            raise ValueError("Observer policy differs from the named arm")
    if min(tr["log_every"], tr["checkpoint_every"]) < 1:
        raise ValueError("Invalid recording intervals")
    dev = protocol["development"]
    if dev["pilot_steps"] != 100 or dev["pilot_batch"] != 512 or dev["zero_slots"] < 4:
        raise ValueError("Expected a 100-update development pilot and mixed-channel zero check")
    ev = protocol["evaluation"]
    if tuple(ev["channels"]) != ("EVA", "ETU") or ev["snrs"] != [12, 16, 20, 24]:
        raise ValueError("Unexpected final-test grid")
    if (ev["backend"], ev["slots"], ev["batch"], ev["bootstrap"], ev["generation"]) != (
            "fft", 2000, 64, 2000, "one_call_per_condition"):
        raise ValueError("Unexpected final-test contract")
    if not {"A0", "A1", "A2"}.issubset(ev["arms"]) or not set(ev["arms"]).issubset(
            {"A0", "A1", "A2", "A1F", "A1A12"}):
        raise ValueError("Unknown or missing inference arms")
    if len(ev["arms"]) != len(set(ev["arms"])):
        raise ValueError("Duplicate inference arms")
    if "A1F" in ev["arms"] and "A1F" not in tr["arms"]:
        raise ValueError("A1F evaluation requires its predeclared training arm")
    for contrast in ev["contrasts"]:
        if len(contrast) != 2 or contrast[0] == contrast[1] or not set(contrast).issubset(ev["arms"]):
            raise ValueError("Invalid paired contrast")
    if ["A1", "A2"] not in ev["contrasts"]:
        raise ValueError("The primary A1-minus-A2 contrast must be predeclared")
    expected_conditions = {condition_key(t, s, ch, snr) for t, s, ch, snr in
                           itertools.product(TAGS, SEEDS, ev["channels"], ev["snrs"])}
    rng = protocol["rng"]
    if set(rng["training_seeds"]) != expected_tasks or set(rng["development_seeds"]) != expected_tasks:
        raise ValueError("Missing training/development RNG seeds")
    if set(rng["final_test_seeds"]) != expected_conditions:
        raise ValueError("Expected all 72 final-test RNG seeds")
    namespaces = rng["namespaces"]
    if set(namespaces) != {"training", "development", "final_test"} or len(set(namespaces.values())) != 3:
        raise ValueError("Training, development, and final-test namespaces must differ")
    all_seeds = [seed for key in ("training_seeds", "development_seeds", "final_test_seeds")
                 for seed in rng[key].values()]
    if any(type(seed) is not int or not 0 <= seed < 2 ** 32 for seed in all_seeds):
        raise ValueError("RNG seeds must be nonnegative 32-bit integers")
    if len(set(all_seeds)) != len(all_seeds):
        raise ValueError("RNG seeds must be distinct within and across namespaces")
    if protocol["environment"]["device"] != "cpu":
        raise ValueError("This runner preserves the existing CPU numerical environment")
    if not Path(protocol["output_root"]).is_absolute():
        raise ValueError("An absolute new output root is required")
    if not protocol.get("historical_cohort_dirs"):
        raise ValueError("Historical cohort identities are required for disjointness checks")


def read_protocol(path, expected_digest):
    path = Path(path).resolve()
    digest = file_hash(path)
    if digest != expected_digest:
        raise ValueError("Protocol digest does not match the locked digest")
    with path.open(encoding="utf-8-sig") as stream:
        protocol = json.load(stream)
    validate_protocol(protocol)
    verify_bindings(protocol)
    return protocol, digest, str(path)


def exclusive_directory(path, protocol):
    target = Path(path).resolve()
    root = Path(protocol["output_root"]).resolve()
    if target == root or root not in target.parents:
        raise ValueError("Use a new task subdirectory inside the protocol output root")
    for protected in (HERE / "ckpt", HERE / "results"):
        if target == protected.resolve() or protected.resolve() in target.parents:
            raise ValueError("Historical checkpoints/results are protected")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.mkdir(exist_ok=False)
    return target


def atomic_json(path, data):
    path = Path(path)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    with temporary.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(data, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def exclusive_torch_save(path, data):
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    with temporary.open("xb") as stream:
        torch.save(data, stream)
        stream.flush()
        os.fsync(stream.fileno())
    if path.exists():
        raise FileExistsError(path)
    os.replace(temporary, path)


def append_record(path, record):
    with Path(path).open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(canonical(record).decode("utf-8") + "\n")


def activation_ranges(model):
    return {name: {"amax": float(module.amax), "calibrated": bool(module.calibrated),
                   "bits": module.bits}
            for name, module in model.named_modules() if isinstance(module, ActQ)}


def model_contract(model, backend):
    ranges = activation_ranges(model)
    return {"wbits": model.qc.wbits, "backend": backend, "qconfig": asdict(model.qc),
            "activation_bits": {name: row["bits"] for name, row in ranges.items()},
            "activation_ranges": ranges, "model_state_sha256": state_hash(model.state_dict())}


def check_finite(state):
    if any(not bool(torch.isfinite(value).all()) for value in state.values()
           if value.is_floating_point() or value.is_complex()):
        raise FloatingPointError("Non-finite model state")


def make_model(tag, state, backend="matrix", output_a12=False):
    cfg = OfdmConfig(nbits=int(tag[1]), long_cp=tag.endswith("longcp"))
    model = StructuredQDCCN(cfg, rank=0, eq_dense=True, slim=True, qc=QConfig())
    model.load_state_dict(state, strict=True)
    model.set_transform_backend(backend)
    if output_a12:
        modules = dict(model.named_modules())
        for name in BOUNDARY:
            modules[name].bits = 12
    model.eval()
    ranges = activation_ranges(model)
    if len(ranges) != 32 or not all(row["calibrated"] for row in ranges.values()):
        raise ValueError("Expected 32 calibrated original activation quantizers")
    if any(row["amax"] < 0 or not np.isfinite(row["amax"]) for row in ranges.values()):
        raise ValueError("Invalid activation range")
    changed = tuple(name for name, row in ranges.items() if row["bits"] != 8)
    if changed != (BOUNDARY if output_a12 else ()):
        raise AssertionError("Unexpected activation-bit intervention")
    check_finite(model.state_dict())
    return model


def load_original(protocol, tag, seed):
    row = protocol["checkpoints"][task_key(tag, seed)]
    if file_hash(row["path"]) != row["sha256"]:
        raise ValueError("Original checkpoint changed")
    state = torch.load(row["path"], map_location="cpu", weights_only=True)
    if not isinstance(state, dict) or len(state) != 88 or not all(isinstance(v, torch.Tensor) for v in state.values()):
        raise ValueError("Expected the original receiver state dictionary, not a training container")
    return state


def train_mode(model, arm):
    if arm not in ("A1", "A1F"):
        raise ValueError("Only continuation arms can enter training")
    model.set_transform_backend("matrix")
    model.train()
    if arm == "A1F":
        for module in model.modules():
            if isinstance(module, ActQ):
                module.eval()


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def capture_rng():
    numpy_state = np.random.get_state()
    return {"torch_cpu": torch.get_rng_state(), "torch_cuda": (
        torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []),
        "python": random.getstate(), "numpy": {"kind": numpy_state[0],
            "keys": torch.from_numpy(numpy_state[1].astype(np.int64)),
            "position": numpy_state[2], "has_gauss": numpy_state[3], "cached_gaussian": numpy_state[4]},
        "generators": {}, "generator_usage": "The unchanged transmitter/channel use the global CPU torch RNG"}


def restore_rng(state):
    torch.set_rng_state(state["torch_cpu"])
    if state["torch_cuda"]:
        if not torch.cuda.is_available():
            raise ValueError("Checkpoint requires unavailable CUDA RNG state")
        torch.cuda.set_rng_state_all(state["torch_cuda"])
    random.setstate(state["python"])
    row = state["numpy"]
    np.random.set_state((row["kind"], row["keys"].numpy().astype(np.uint32),
                         row["position"], row["has_gauss"], row["cached_gaussian"]))


def channels_for(cfg):
    channels = {name: ChannelTorch(cfg, torch.device("cpu"), model=name) for name in CHANNELS}
    if any(channel.pdp_mode != "3gpp_db_to_linear_power" for name, channel in channels.items() if name != "flat"):
        raise ValueError("Corrected-PDP training is required")
    return channels


def draw_training(tx, channels, batch):
    transmitted, labels, _ = tx.transmit(batch)
    snr = sample_mixed_snr(batch, torch.device("cpu"))
    received = torch.empty_like(transmitted)
    taps_by_channel = {}
    for index, name in enumerate(CHANNELS):
        selected = torch.arange(index, batch, len(CHANNELS))
        received[selected], taps = apply_fading_awgn(channels[name], transmitted[selected], snr[selected])
        taps_by_channel[name] = tensor_hash(taps)
    hashes = {"transmitted": tensor_hash(transmitted), "labels": tensor_hash(labels),
              "snr": tensor_hash(snr), "received": tensor_hash(received),
              "taps": hashlib.sha256(canonical(taps_by_channel)).hexdigest()}
    return received, labels, hashes, taps_by_channel


def optimization_step(model, optimizer, scheduler, received, labels):
    soft = model(received)
    if not bool(torch.isfinite(soft).all()):
        raise FloatingPointError("Non-finite training output")
    loss = ce_loss(soft, labels)
    if not bool(torch.isfinite(loss)):
        raise FloatingPointError("Non-finite training loss")
    optimizer.zero_grad()
    loss.backward()
    if any(parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all())
           for parameter in model.parameters()):
        raise FloatingPointError("Non-finite training gradient")
    optimizer.step()
    scheduler.step()
    check_finite(model.state_dict())
    return float(loss.detach()), float(ber(soft.detach(), labels))


def peak_memory_bytes():
    if os.name == "nt":
        class Counters(ctypes.Structure):
            _fields_ = [("cb", ctypes.c_ulong), ("PageFaultCount", ctypes.c_ulong)] + [
                (name, ctypes.c_size_t) for name in ("PeakWorkingSetSize", "WorkingSetSize",
                "QuotaPeakPagedPoolUsage", "QuotaPagedPoolUsage", "QuotaPeakNonPagedPoolUsage",
                "QuotaNonPagedPoolUsage", "PagefileUsage", "PeakPagefileUsage")]
        counters = Counters()
        counters.cb = ctypes.sizeof(counters)
        current_process = ctypes.windll.kernel32.GetCurrentProcess
        current_process.restype = ctypes.c_void_p
        function = ctypes.windll.psapi.GetProcessMemoryInfo
        function.argtypes = [ctypes.c_void_p, ctypes.POINTER(Counters), ctypes.c_ulong]
        if function(current_process(), ctypes.byref(counters), counters.cb):
            return int(counters.PeakWorkingSetSize)
        return None
    try:
        import resource
        value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return int(value if sys.platform == "darwin" else value * 1024)
    except ImportError:
        return None


def base_manifest(protocol, digest, protocol_path, stage):
    return {"schema_version": 1, "stage": stage, "status": "running",
            "protocol_path": protocol_path, "protocol_sha256": digest,
            "experiment_id": protocol["experiment_id"], "source_hashes": protocol["source_hashes"],
            "environment": environment(), "original_checkpoints": protocol["checkpoints"],
            "final_test_used": stage == "evaluate"}


def run_zero(protocol, digest, protocol_path, output, tag=None, seed=None):
    from eval_struct_qat import build_model

    manifest = base_manifest(protocol, digest, protocol_path, "zero")
    manifest["conditions"] = []
    atomic_json(output / "manifest.json", manifest)
    try:
        for this_tag in ([tag] if tag else TAGS):
            for this_seed in ([seed] if seed else SEEDS):
                state = load_original(protocol, this_tag, this_seed)
                model = make_model(this_tag, state, "fft")
                cfg = model.cfg
                tx, channels = OfdmTxTorch(cfg, torch.device("cpu")), channels_for(cfg)
                reference = build_model("qat_s_r0_slim_w8a8", cfg, this_tag, tx, channels,
                                        checkpoint_suffix=f"_pdpfix_seed{this_seed}")
                reference.set_transform_backend("fft")
                dev_seed = protocol["rng"]["development_seeds"][task_key(this_tag, this_seed)]
                seed_all(dev_seed)
                received, labels, hashes, _ = draw_training(tx, channels, protocol["development"]["zero_slots"])
                row = {"tag": this_tag, "training_seed": this_seed, "development_rng_seed": dev_seed,
                       "reference_loader": "eval_struct_qat.build_model, original QAT, FFT backend",
                       "stimulus_sha256": hashes, "arms": {}}
                for arm in protocol["evaluation"]["arms"]:
                    candidate = make_model(this_tag, state, "fft", arm in ("A2", "A1A12"))
                    reference.eq.m_back.q_out.qr.bits = reference.eq.m_back.q_out.qi.bits = (
                        12 if arm in ("A2", "A1A12") else 8)
                    with torch.inference_mode():
                        expected, actual = reference(received), candidate(received)
                    if not torch.equal(actual, expected) or state_hash(candidate.state_dict()) != state_hash(state):
                        raise AssertionError("Zero-update inference does not match the original model: " + arm)
                    row["arms"][arm] = {"max_absolute_difference": float((actual - expected).abs().max()),
                                        "model_contract": model_contract(candidate, "fft")}
                manifest["conditions"].append(row)
        verify_bindings(protocol)
        manifest.update(status="complete", all_source_and_checkpoint_hashes_unchanged=True)
    except BaseException as exc:
        manifest.update(status="failed_or_interrupted", error=repr(exc))
        atomic_json(output / "manifest.json", manifest)
        raise
    atomic_json(output / "manifest.json", manifest)
    return manifest


def full_checkpoint(model, optimizer, scheduler, manifest):
    row = {key: copy.deepcopy(value) for key, value in manifest.items() if key not in ("checkpoints", "status")}
    row.update(kind="continued_qat_full_state", model=copy.deepcopy(model.state_dict()),
               optimizer=copy.deepcopy(optimizer.state_dict()), scheduler=copy.deepcopy(scheduler.state_dict()),
               rng=capture_rng(), final_activation_ranges=activation_ranges(model),
               model_contract=model_contract(model, "matrix"), peak_process_memory_bytes=peak_memory_bytes())
    return row


def resume_logs(resume_path, output, checkpoint):
    previous_root = resume_path.parent.parent if resume_path.parent.name == "checkpoints" else resume_path.parent
    chain, seen = CHAIN_INITIAL, 0
    source = previous_root / "stimuli.jsonl"
    with source.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if row["step"] > checkpoint["step"]:
                break
            if row["step"] != seen + 1:
                raise ValueError("Resume stimulus log is not contiguous")
            chain = advance_chain(chain, row["step"], row["stimulus_sha256"])
            if chain != row["chain_sha256"]:
                raise ValueError("Resume stimulus chain is invalid")
            append_record(output / "stimuli.jsonl", row)
            seen += 1
    if seen != checkpoint["step"] or chain != checkpoint["training_stimulus_chain_sha256"]:
        raise ValueError("Resume checkpoint and stimulus log disagree")
    with (previous_root / "training.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if row["step"] <= seen:
                append_record(output / "training.jsonl", row)


def run_training(protocol, digest, protocol_path, output, tag, seed, arm, pilot=False, resume=None):
    if arm not in protocol["training"]["arms"]:
        raise ValueError("Training arm was not predeclared")
    if pilot and resume:
        raise ValueError("Development pilots cannot resume formal training")
    stage = "pilot" if pilot else "train"
    setting = protocol["training"]
    steps = protocol["development"]["pilot_steps"] if pilot else setting["steps"]
    batch = protocol["development"]["pilot_batch"] if pilot else setting["batch"]
    state = load_original(protocol, tag, seed)
    model = make_model(tag, state)
    initial = activation_ranges(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=setting["lr"], weight_decay=setting["weight_decay"])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=setting["step_size"], gamma=setting["gamma"])
    tx, channels = OfdmTxTorch(model.cfg, torch.device("cpu")), channels_for(model.cfg)
    rng_group = "development_seeds" if pilot else "training_seeds"
    rng_seed = protocol["rng"][rng_group][task_key(tag, seed)]
    manifest = base_manifest(protocol, digest, protocol_path, stage)
    manifest.update(tag=tag, training_seed=seed, arm=arm, task_rng_seed=rng_seed,
                    rng_namespace=protocol["rng"]["namespaces"]["development" if pilot else "training"],
                    step=0, expected_steps=steps, batch=batch, training_slots=0,
                    observer_mode=setting["observer_modes"][arm], selection="final",
                    original_checkpoint=protocol["checkpoints"][task_key(tag, seed)],
                    initial_activation_ranges=initial, training_stimulus_chain_sha256=CHAIN_INITIAL,
                    checkpoints=[], elapsed_seconds=0.0, optimization_seconds=0.0,
                    scope="Development cost/numerical pilot, no final test" if pilot else
                    "Warm-start selected QAT weights; new Adam/scheduler; fixed final checkpoint")
    started, old_elapsed = time.perf_counter(), 0.0
    if resume:
        resume_path = Path(resume).resolve()
        checkpoint = torch.load(resume_path, map_location="cpu", weights_only=True)
        expected = {"kind": "continued_qat_full_state", "protocol_sha256": digest,
                    "tag": tag, "training_seed": seed, "arm": arm, "stage": "train",
                    "expected_steps": steps, "batch": batch, "task_rng_seed": rng_seed}
        if any(checkpoint.get(key) != value for key, value in expected.items()) or not 0 <= checkpoint["step"] < steps:
            raise ValueError("Resume state does not belong to this unfinished fixed-budget task")
        model.load_state_dict(checkpoint["model"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        if checkpoint["initial_activation_ranges"] != initial:
            raise ValueError("Resume state has a different original calibration")
        resume_logs(resume_path, output, checkpoint)
        for key in ("step", "training_slots", "training_stimulus_chain_sha256", "optimization_seconds"):
            manifest[key] = checkpoint[key]
        old_elapsed = checkpoint["elapsed_seconds"]
        manifest["resume_from"] = {"path": str(resume_path), "sha256": file_hash(resume_path),
                                   "step": checkpoint["step"]}
        restore_rng(checkpoint["rng"])
    else:
        seed_all(rng_seed)
    train_mode(model, arm)
    atomic_json(output / "manifest.json", manifest)
    try:
        for step in range(manifest["step"] + 1, steps + 1):
            received, labels, hashes, taps = draw_training(tx, channels, batch)
            optimization_start = time.perf_counter()
            loss, batch_ber = optimization_step(model, optimizer, scheduler, received, labels)
            manifest["optimization_seconds"] += time.perf_counter() - optimization_start
            chain = advance_chain(manifest["training_stimulus_chain_sha256"], step, hashes)
            append_record(output / "stimuli.jsonl", {"step": step, "training_rng_seed": rng_seed,
                "stimulus_sha256": hashes, "taps_by_channel": taps, "chain_sha256": chain})
            manifest.update(step=step, training_slots=step * batch,
                            training_stimulus_chain_sha256=chain,
                            elapsed_seconds=old_elapsed + time.perf_counter() - started)
            if step % setting["log_every"] == 0 or step == steps:
                current = activation_ranges(model)
                if arm == "A1F" and current != initial:
                    raise AssertionError("Frozen activation ranges changed")
                if any(current[name]["amax"] < initial[name]["amax"] for name in initial):
                    raise AssertionError("Original running maxima unexpectedly decreased")
                row = {"step": step, "loss": loss, "train_batch_ber": batch_ber,
                       "lr": optimizer.param_groups[0]["lr"], "elapsed_seconds": manifest["elapsed_seconds"],
                       "training_slots": manifest["training_slots"], "training_stimulus_chain_sha256": chain,
                       "activation_ranges": current, "peak_process_memory_bytes": peak_memory_bytes()}
                append_record(output / "training.jsonl", row)
                print(f"{tag} seed{seed} {arm} {stage} {step}/{steps}: loss={loss:.6g} "
                      f"train-BER={batch_ber:.6g} elapsed={manifest['elapsed_seconds']:.1f}s", flush=True)
                atomic_json(output / "manifest.json", manifest)
            if step % setting["checkpoint_every"] == 0 and step < steps:
                path = output / "checkpoints" / f"step_{step:06d}.pt"
                exclusive_torch_save(path, full_checkpoint(model, optimizer, scheduler, manifest))
                manifest["checkpoints"].append({"path": str(path), "sha256": file_hash(path), "step": step})
        model.eval()
        current = activation_ranges(model)
        if arm == "A1F" and current != initial:
            raise AssertionError("Frozen activation ranges changed")
        verify_bindings(protocol)
        manifest.update(final_activation_ranges=current, model_contract=model_contract(model, "matrix"),
                        peak_process_memory_bytes=peak_memory_bytes(),
                        seconds_per_update=manifest["elapsed_seconds"] / steps)
        path = output / ("pilot_final.pt" if pilot else "final.pt")
        exclusive_torch_save(path, full_checkpoint(model, optimizer, scheduler, manifest))
        manifest.update(status="complete", final_checkpoint={"path": str(path), "sha256": file_hash(path)},
                        all_source_and_checkpoint_hashes_unchanged=True)
    except BaseException as exc:
        manifest.update(status="failed_or_interrupted", error=repr(exc),
                        elapsed_seconds=old_elapsed + time.perf_counter() - started)
        atomic_json(output / "manifest.json", manifest)
        raise
    atomic_json(output / "manifest.json", manifest)
    return manifest


def load_final(protocol, digest, training_root, tag, seed, arm):
    directory = Path(training_root) / task_key(tag, seed) / arm
    with (directory / "manifest.json").open(encoding="utf-8") as stream:
        manifest = json.load(stream)
    expected = {"status": "complete", "stage": "train", "protocol_sha256": digest,
                "tag": tag, "training_seed": seed, "arm": arm,
                "step": protocol["training"]["steps"], "selection": "final", "final_test_used": False}
    if any(manifest.get(key) != value for key, value in expected.items()):
        raise ValueError("Formal fixed-budget final checkpoint is not complete: " + str(directory))
    path = directory / "final.pt"
    if Path(manifest["final_checkpoint"]["path"]).resolve() != path.resolve() or file_hash(path) != manifest["final_checkpoint"]["sha256"]:
        raise ValueError("Final checkpoint path/digest differs from completed training manifest")
    full = torch.load(path, map_location="cpu", weights_only=True)
    for key in ("stage", "protocol_sha256", "tag", "training_seed", "arm", "step", "selection"):
        if full.get(key) != expected[key]:
            raise ValueError("Full checkpoint identity differs: " + key)
    if full.get("kind") != "continued_qat_full_state":
        raise ValueError("Expected complete training state")
    return full["model"], {"path": str(path.resolve()), "sha256": file_hash(path),
                            "training_manifest_sha256": file_hash(directory / "manifest.json")}


def noise_residual(transmitted, received, taps, channel):
    slots = transmitted.shape[0]
    x = transmitted.reshape(slots, -1)
    if channel.L == 1:
        clean = x * taps
    else:
        nfft = 1 << (x.shape[1] + channel.L - 2).bit_length()
        clean = torch.fft.ifft(torch.fft.fft(x, n=nfft, dim=1) *
                                torch.fft.fft(taps, n=nfft, dim=1), dim=1)[:, :x.shape[1]]
    return received.reshape(slots, -1) - clean


def paired_counts(predictions, labels, contrasts, bootstrap, seed):
    truth = labels.bool().reshape(labels.shape[0], -1)
    arrays = {"labels": truth.numpy()}
    arms, summaries = {}, {}
    errors = {arm: prediction.reshape_as(truth) != truth for arm, prediction in predictions.items()}
    for arm, error in errors.items():
        slot_errors = error.sum(1).numpy()
        arrays[arm + "__pred"] = predictions[arm].reshape_as(truth).numpy()
        arrays[arm + "__slot_errors"] = slot_errors
        arms[arm] = {"error_bits": int(slot_errors.sum()), "ber": float(error.float().mean()),
                     "slots": len(truth), "bits_per_slot": truth.shape[1]}
        arms[arm]["ber"] = int(slot_errors.sum()) / truth.numel()
    for candidate, reference in contrasts:
        key = candidate + "_minus_" + reference
        corrected = (errors[reference] & ~errors[candidate]).sum(1).numpy()
        harmed = (~errors[reference] & errors[candidate]).sum(1).numpy()
        flips = (predictions[candidate].reshape_as(truth) != predictions[reference].reshape_as(truth)).sum(1).numpy()
        if not np.array_equal(flips, corrected + harmed) or not np.array_equal(
                arrays[candidate + "__slot_errors"] - arrays[reference + "__slot_errors"], harmed - corrected):
            raise AssertionError("Paired decision conservation failed")
        for suffix, data in (("corrected", corrected), ("harmed", harmed), ("flips", flips)):
            arrays[key + "__" + suffix] = data
        summaries[key] = summarize_paired_slots(arrays[candidate + "__slot_errors"],
            arrays[reference + "__slot_errors"], truth.shape[1], bootstrap_reps=bootstrap, bootstrap_seed=seed)
        summaries[key].update(corrected_decisions=int(corrected.sum()), harmed_decisions=int(harmed.sum()),
                              changed_decisions=int(flips.sum()))
    return arrays, arms, summaries


def infer_predictions(model, received, batch):
    if model.training or any(module.training for module in model.modules() if isinstance(module, ActQ)):
        raise ValueError("Inference must freeze the model and all observers")
    predictions = []
    with torch.inference_mode():
        for start in range(0, len(received), batch):
            probabilities = model(received[start:start + batch])
            if not bool(torch.isfinite(probabilities).all()):
                raise FloatingPointError("Non-finite inference")
            predictions.append(probabilities[..., 1] > .5)
    return torch.cat(predictions)


def historical_received_hashes(protocol):
    received = set()
    for directory in protocol["historical_cohort_dirs"]:
        paths = list(Path(directory).glob("*_seed*dB.json"))
        if not paths:
            raise ValueError("Historical cohort directory is empty: " + str(directory))
        for path in paths:
            with path.open(encoding="utf-8") as stream:
                row = json.load(stream)
            received.add(row["stimulus_sha256"]["received"])
    return received


def run_evaluation(protocol, digest, protocol_path, output, training_root, authorized=False):
    if not authorized:
        raise ValueError("Final-test evaluation requires --authorize-final-test")
    ev = protocol["evaluation"]
    # Check all required tasks BEFORE generating even one final-test sample.
    finals = {(tag, seed, arm): load_final(protocol, digest, training_root, tag, seed, arm)
              for tag, seed, arm in itertools.product(TAGS, SEEDS, protocol["training"]["arms"])}
    old_hashes = historical_received_hashes(protocol)
    training_hashes = set()
    for tag, seed, arm in itertools.product(TAGS, SEEDS, protocol["training"]["arms"]):
        with (Path(training_root) / task_key(tag, seed) / arm / "stimuli.jsonl").open(encoding="utf-8") as stream:
            training_hashes.update(json.loads(line)["stimulus_sha256"]["received"] for line in stream)
    manifest = base_manifest(protocol, digest, protocol_path, "evaluate")
    manifest.update(args={"tags": protocol["tags"], "seeds": protocol["seeds"],
        "channels": ev["channels"], "snrs": ev["snrs"], "slots": ev["slots"],
        "batch": ev["batch"], "bootstrap": ev["bootstrap"], "training_root": str(Path(training_root).resolve())},
        conditions=[], training_checkpoints={
        task_key(tag, seed) + "_" + arm: binding for (tag, seed, arm), (_, binding) in finals.items()},
        historical_received_hashes_checked=len(old_hashes), contrasts=ev["contrasts"],
        variants_fixed_before_run=ev["arms"], unique_slots=0, unique_bits=0,
        statistics="Paired-slot conditional percentile intervals; seed variation analyzed separately")
    atomic_json(output / "manifest.json", manifest)
    seen, started = set(), time.perf_counter()
    try:
        for tag, seed in itertools.product(TAGS, SEEDS):
            original = load_original(protocol, tag, seed)
            models, bindings = {}, {}
            for arm in ev["arms"]:
                base_arm = "A1" if arm == "A1A12" else arm
                state, binding = (finals[(tag, seed, base_arm)] if arm in ("A1", "A1F", "A1A12")
                                  else (original, protocol["checkpoints"][task_key(tag, seed)]))
                models[arm] = make_model(tag, state, "fft", arm in ("A2", "A1A12"))
                bindings[arm] = binding["sha256"]
            contracts = {arm: model_contract(model, "fft") for arm, model in models.items()}
            if contracts["A0"]["model_state_sha256"] != contracts["A2"]["model_state_sha256"]:
                raise AssertionError("A2 changed original model state")
            if "A1A12" in models and contracts["A1"]["model_state_sha256"] != contracts["A1A12"]["model_state_sha256"]:
                raise AssertionError("A1A12 changed continued model state")
            cfg = models["A0"].cfg
            tx = OfdmTxTorch(cfg, torch.device("cpu"))
            for channel_name, snr in itertools.product(ev["channels"], ev["snrs"]):
                name = condition_key(tag, seed, channel_name, snr)
                point_seed = protocol["rng"]["final_test_seeds"][name]
                seed_all(point_seed)
                channel = ChannelTorch(cfg, torch.device("cpu"), model=channel_name)
                if channel.pdp_mode != "3gpp_db_to_linear_power":
                    raise AssertionError("Expected corrected PDP")
                with torch.inference_mode():
                    transmitted, labels, _ = tx.transmit(ev["slots"])
                    received, taps = apply_fading_awgn(channel, transmitted, torch.full((ev["slots"],), float(snr)))
                    noise = noise_residual(transmitted, received, taps, channel)
                    hashes = {key: tensor_hash(value) for key, value in (("transmitted", transmitted),
                        ("labels", labels), ("received", received), ("taps", taps), ("noise_residual", noise))}
                    if hashes["received"] in old_hashes | training_hashes | seen:
                        raise AssertionError("Final-test stimulus duplicates training/history/another condition")
                    seen.add(hashes["received"])
                    predictions = {}
                    for arm, model in models.items():
                        predictions[arm] = infer_predictions(model, received, ev["batch"])
                    arrays, summaries, contrasts = paired_counts(predictions, labels, ev["contrasts"],
                        ev["bootstrap"], point_seed + ev["bootstrap_seed_offset"])
                if tensor_hash(received) != hashes["received"]:
                    raise AssertionError("Received tensor mutated during paired inference")
                path = output / (name + ".npz")
                with path.open("xb") as stream:
                    np.savez_compressed(stream, **arrays)
                row = {"tag": tag, "training_seed": seed, "channel": channel_name, "snr_db": snr,
                       "slots": ev["slots"], "bits_per_slot": cfg.D_slot * cfg.m, "point_seed": point_seed,
                       "stimulus_sha256": hashes, "pdp_mode": channel.pdp_mode, "arms": summaries,
                       "contrasts": contrasts, "slot_data": path.name, "slot_data_sha256": file_hash(path),
                       "checkpoint_sha256": bindings, "arm_contracts": contracts,
                       "protocol_sha256": digest, "protocol_path": protocol_path}
                atomic_json(output / (name + ".json"), row)
                manifest["conditions"].append(name + ".json")
                manifest["unique_slots"] += ev["slots"]
                manifest["unique_bits"] += ev["slots"] * cfg.D_slot * cfg.m
                atomic_json(output / "manifest.json", manifest)
                print(name + ": " + ", ".join(arm + " BER=" + f"{value['ber']:.7g}" for arm, value in summaries.items()), flush=True)
            for arm, model in models.items():
                if model_contract(model, "fft") != contracts[arm]:
                    raise AssertionError("Inference changed frozen model state: " + arm)
        verify_bindings(protocol)
        for binding in manifest["training_checkpoints"].values():
            if file_hash(binding["path"]) != binding["sha256"]:
                raise AssertionError("Training checkpoint changed during final test")
        manifest.update(status="complete", elapsed_seconds=time.perf_counter() - started,
                        all_source_and_checkpoint_hashes_unchanged=True,
                        received_cohorts_disjoint_from_history_and_training=True)
    except BaseException as exc:
        manifest.update(status="failed_or_interrupted", error=repr(exc))
        atomic_json(output / "manifest.json", manifest)
        raise
    atomic_json(output / "manifest.json", manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("zero", "pilot", "train", "evaluate"):
        sub = commands.add_parser(name)
        sub.add_argument("--protocol", type=Path, required=True)
        sub.add_argument("--protocol-sha256", required=True)
        sub.add_argument("--output", type=Path, required=True)
        if name != "evaluate":
            sub.add_argument("--tag", choices=TAGS, required=name != "zero")
            sub.add_argument("--seed", choices=SEEDS, type=int, required=name != "zero")
        if name in ("pilot", "train"):
            sub.add_argument("--arm", choices=("A1", "A1F"), default="A1")
        if name == "train":
            sub.add_argument("--resume", type=Path)
        if name == "evaluate":
            sub.add_argument("--training-root", type=Path, required=True)
            sub.add_argument("--authorize-final-test", action="store_true")
    args = parser.parse_args()
    protocol, digest, protocol_path = read_protocol(args.protocol, args.protocol_sha256)
    torch.set_num_threads(protocol["environment"]["threads"])
    torch.set_num_interop_threads(1)
    actual = environment()
    for key, expected in protocol["environment"].items():
        if key in actual and actual[key] != expected:
            parser.error("Runtime environment differs from protocol: " + key)
    if args.command == "evaluate" and not args.authorize_final_test:
        parser.error("Opening final test requires --authorize-final-test")
    output = exclusive_directory(args.output, protocol)
    if args.command == "zero":
        result = run_zero(protocol, digest, protocol_path, output, args.tag, args.seed)
    elif args.command in ("pilot", "train"):
        result = run_training(protocol, digest, protocol_path, output, args.tag, args.seed, args.arm,
                              pilot=args.command == "pilot", resume=getattr(args, "resume", None))
    else:
        result = run_evaluation(protocol, digest, protocol_path, output, args.training_root, args.authorize_final_test)
    print(json.dumps({"status": result["status"], "stage": result["stage"],
                      "output": str(output), "final_test_used": result["final_test_used"]}), flush=True)


if __name__ == "__main__":
    main()
