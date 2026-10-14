"""Descriptive summaries of the locked continuation experiment, after audit.

Reads existing completed results; generates no stimuli and trains no model.
All 72 conditions, five inference arms, 18 continuation runs, and six planned
contrasts are retained. Seed SD and conditional slot intervals are distinct.
Only observed 12--24 dB crossings are interpolated; censoring is explicit.
"""
import argparse
import csv
import hashlib
import itertools
import json
import math
import platform
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
TAGS = ("b1_longcp", "b2_longcp", "b2_shortcp")
SEEDS = (1, 2, 3)
ARMS = ("A0", "A1", "A2", "A1F", "A1A12")
CONTRASTS = (("A1", "A2"), ("A1", "A0"), ("A2", "A0"),
             ("A1F", "A0"), ("A1F", "A1"), ("A1A12", "A1"))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    with Path(path).open(encoding="utf-8-sig") as stream:
        return json.load(stream)


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def write_csv(path, rows):
    require(bool(rows), "Cannot export an empty source-data table")
    fields = list(rows[0])
    require(all(list(row) == fields for row in rows), "Inconsistent CSV field order")
    with Path(path).open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def key(tag, seed, channel=None, snr=None):
    result = f"{tag}_seed{seed}"
    return result if channel is None else result + f"_{channel}_{snr:g}dB"


def relative(delta, reference):
    return None if reference == 0 else 100.0 * delta / reference


def observed_threshold(snrs, bers, target=.02):
    snrs, raw = np.asarray(snrs, dtype=float), np.asarray(bers, dtype=float)
    require(len(snrs) == len(raw) >= 2 and bool(np.all(np.diff(snrs) > 0)), "Invalid SNR curve")
    require(bool(np.all(np.isfinite(raw))) and bool(np.all((raw >= 0) & (raw <= 1))), "Invalid BER curve")
    envelope = np.minimum.accumulate(raw)
    result = {"target_ber": target, "threshold_db": None, "status": "",
              "lower_bound_db": None, "upper_bound_db": None,
              "raw_nonmonotone_increases": int(np.sum(np.diff(raw) > 0)),
              "interpolation": "running-minimum envelope; linear SNR versus log10 BER; no extrapolation"}
    if envelope[0] <= target:
        result.update(status="left_censored", upper_bound_db=float(snrs[0]))
        return result
    if envelope[-1] > target:
        result.update(status="right_censored", lower_bound_db=float(snrs[-1]))
        return result
    for index in range(len(snrs) - 1):
        if envelope[index] > target >= envelope[index + 1]:
            result.update(lower_bound_db=float(snrs[index]), upper_bound_db=float(snrs[index + 1]))
            if envelope[index + 1] == 0:
                result.update(status="zero_endpoint_not_log_interpolable")
                return result
            first, last = math.log10(envelope[index]), math.log10(envelope[index + 1])
            fraction = (math.log10(target) - first) / (last - first)
            result.update(status="observed_crossing",
                          threshold_db=float(snrs[index] + fraction * (snrs[index + 1] - snrs[index])))
            return result
    raise AssertionError("An observed crossing was not resolved")


def checked_inputs(protocol, protocol_sha, run, audit):
    manifest_path = run / "manifest.json"
    manifest = read_json(manifest_path)
    require(manifest["status"] == "complete" and manifest["stage"] == "evaluate", "Require completed final evaluation")
    require(manifest["protocol_sha256"] == protocol_sha, "Evaluation protocol changed")
    require(manifest["unique_slots"] == 144000 and manifest["unique_bits"] == 76800000, "Unexpected unique stimulus budget")
    report = read_json(audit)
    require(report["status"] == "pass" and not report.get("smoke_not_for_publication", True), "Require a formal independent audit pass")
    require(report["protocol_sha256"] == protocol_sha, "Audit protocol differs")
    require(report["evaluation_manifest_sha256"] == sha(manifest_path), "Audit does not bind this completed evaluation")
    for field in ("all_input_hashes_regenerated", "all_raw_predictions_and_paired_counts_rebuilt",
                  "all_conditional_slot_bootstrap_intervals_rebuilt"):
        require(report.get(field) is True, "Independent audit is incomplete: " + field)
    require(isinstance(report.get("input_bindings"), list) and bool(report["input_bindings"]),
            "Independent audit must bind all inspected artifacts by SHA-256")
    for binding in report["input_bindings"]:
        require(sha(binding["path"]) == binding["sha256"], "Artifact changed after independent audit: " + binding["path"])
    require(tuple(protocol["tags"]) == TAGS and tuple(protocol["seeds"]) == SEEDS, "Unexpected configuration/seed cohort")
    require(tuple(protocol["evaluation"]["arms"]) == ARMS, "Expected all five planned arms")
    require(tuple(map(tuple, protocol["evaluation"]["contrasts"])) == CONTRASTS, "Expected all six planned contrasts")
    expected = {key(tag, seed, channel, snr) + ".json" for tag, seed, channel, snr in itertools.product(
        TAGS, SEEDS, ("EVA", "ETU"), (12, 16, 20, 24))}
    require(len(manifest["conditions"]) == 72 and set(manifest["conditions"]) == expected, "Missing/duplicate final conditions")
    for name, row in protocol["checkpoints"].items():
        require(sha(row["path"]) == row["sha256"], "Original checkpoint changed: " + name)
    return manifest, report


def condition_tables(run, manifest, protocol_sha):
    arm_rows, contrast_rows, bindings, lookup = [], [], [], {}
    for name in sorted(manifest["conditions"]):
        path = run / name
        row = read_json(path)
        require(row["protocol_sha256"] == protocol_sha and row["slots"] == 2000, "Unexpected condition protocol/budget")
        expected_bits = 320 if row["tag"] == "b1_longcp" else 640
        require(row["bits_per_slot"] == expected_bits, "Wrong modulation denominator")
        identity = (row["tag"], row["training_seed"], row["channel"], row["snr_db"])
        require(identity not in lookup, "Duplicate condition identity")
        lookup[identity] = row
        npz = (run / row["slot_data"]).resolve()
        require(run.resolve() in npz.parents and sha(npz) == row["slot_data_sha256"], "NPZ path/hash differs")
        denominator = row["slots"] * expected_bits
        with np.load(npz, allow_pickle=False) as arrays:
            truth = arrays["labels"]
            require(truth.shape == (2000, expected_bits) and truth.dtype == np.bool_, "Wrong stored label shape/type")
            error_masks = {}
            for arm in ARMS:
                prediction = arrays[arm + "__pred"]
                require(prediction.shape == truth.shape and prediction.dtype == np.bool_, "Wrong prediction shape/type")
                error_masks[arm] = prediction != truth
                counts = error_masks[arm].sum(1)
                require(np.array_equal(counts, arrays[arm + "__slot_errors"]), "Slot errors disagree with raw decisions")
                error_bits = int(counts.sum())
                require(error_bits == row["arms"][arm]["error_bits"], "Condition error count differs")
                require(abs(row["arms"][arm]["ber"] - error_bits / denominator) <= 1e-15, "Condition BER differs")
                arm_rows.append({"tag": row["tag"], "training_seed": row["training_seed"],
                    "channel": row["channel"], "snr_db": row["snr_db"], "arm": arm,
                    "slots": row["slots"], "bits_per_slot": expected_bits,
                    "unique_bits": denominator, "error_bits": error_bits, "ber": error_bits / denominator,
                    "checkpoint_sha256": row["checkpoint_sha256"][arm],
                    "received_sha256": row["stimulus_sha256"]["received"]})
            for candidate, reference in CONTRASTS:
                name_key = candidate + "_minus_" + reference
                pair = row["contrasts"][name_key]
                corrected = (error_masks[reference] & ~error_masks[candidate]).sum(1)
                harmed = (~error_masks[reference] & error_masks[candidate]).sum(1)
                flips = corrected + harmed
                for suffix, value in (("corrected", corrected), ("harmed", harmed), ("flips", flips)):
                    require(np.array_equal(value, arrays[name_key + "__" + suffix]), "Paired bit counts differ")
                delta = float((harmed.astype(float) - corrected).mean() / expected_bits)
                require(abs(delta - pair["candidate_minus_reference_ber"]) <= 1e-15, "Pair BER difference differs")
                interval = pair["conditional_slot_bootstrap_95pct_ci"]
                require(pair["bootstrap_reps"] == 2000 and len(interval) == 2, "Missing conditional interval")
                baseline = row["arms"][reference]["ber"]
                contrast_rows.append({"tag": row["tag"], "training_seed": row["training_seed"],
                    "channel": row["channel"], "snr_db": row["snr_db"], "contrast": name_key,
                    "candidate": candidate, "reference": reference,
                    "candidate_ber": row["arms"][candidate]["ber"], "reference_ber": baseline,
                    "delta_ber": delta, "relative_change_percent": relative(delta, baseline),
                    "corrected_decisions": int(corrected.sum()), "harmed_decisions": int(harmed.sum()),
                    "changed_decisions": int(flips.sum()), "conditional_slot_ci_low": interval[0],
                    "conditional_slot_ci_high": interval[1], "slots": row["slots"],
                    "bits_per_slot": expected_bits, "bootstrap_reps": 2000})
        bindings.extend({"path": str(p), "sha256": sha(p)} for p in (path, npz))
    require(len(arm_rows) == 360 and len(contrast_rows) == 432, "Incomplete source tables")
    return arm_rows, contrast_rows, lookup, bindings


def seed_tables(lookup):
    arms, contrasts = [], []
    for tag, seed in itertools.product(TAGS, SEEDS):
        selected = [lookup[(tag, seed, channel, snr)] for channel, snr in itertools.product(("EVA", "ETU"), (12, 16, 20, 24))]
        mean_ber = {arm: float(np.mean([row["arms"][arm]["ber"] for row in selected])) for arm in ARMS}
        for arm in ARMS:
            delta = mean_ber[arm] - mean_ber["A0"]
            arms.append({"tag": tag, "training_seed": seed, "arm": arm, "conditions": 8,
                "grid_mean_ber": mean_ber[arm], "a0_grid_mean_ber": mean_ber["A0"],
                "delta_from_a0_ber": delta, "relative_change_from_a0_percent": relative(delta, mean_ber["A0"]),
                "error_bits": sum(row["arms"][arm]["error_bits"] for row in selected),
                "unique_slots": 16000, "unique_bits": 16000 * (320 if tag == "b1_longcp" else 640)})
        for candidate, reference in CONTRASTS:
            name = candidate + "_minus_" + reference
            delta = mean_ber[candidate] - mean_ber[reference]
            contrast = [row["contrasts"][name]["candidate_minus_reference_ber"] for row in selected]
            require(abs(delta - float(np.mean(contrast))) < 1e-15, "Grid pairing identity failed")
            contrasts.append({"tag": tag, "training_seed": seed, "contrast": name,
                "candidate": candidate, "reference": reference, "conditions": 8,
                "candidate_grid_mean_ber": mean_ber[candidate], "reference_grid_mean_ber": mean_ber[reference],
                "delta_ber": delta, "relative_change_percent": relative(delta, mean_ber[reference]),
                "corrected_decisions": sum(row["contrasts"][name]["corrected_decisions"] for row in selected),
                "harmed_decisions": sum(row["contrasts"][name]["harmed_decisions"] for row in selected),
                "changed_decisions": sum(row["contrasts"][name]["changed_decisions"] for row in selected)})
    return arms, contrasts


def condition_seed_tables(arm_points, contrast_points):
    """Keep condition-specific seed dispersion separate from the grid mean."""
    arm_summary, contrast_summary = [], []
    for tag, channel, snr, arm in itertools.product(TAGS, ("EVA", "ETU"), (12, 16, 20, 24), ARMS):
        rows = sorted((row for row in arm_points if
                      (row["tag"], row["channel"], row["snr_db"], row["arm"]) == (tag, channel, snr, arm)),
                      key=lambda row: row["training_seed"])
        require([row["training_seed"] for row in rows] == list(SEEDS), "Missing condition/arm seed")
        values = [row["ber"] for row in rows]
        arm_summary.append({"tag": tag, "channel": channel, "snr_db": snr, "arm": arm,
            "training_seeds": 3, "seed1_ber": values[0], "seed2_ber": values[1], "seed3_ber": values[2],
            "mean_ber": float(np.mean(values)), "sample_sd_ber": float(np.std(values, ddof=1))})
    for tag, channel, snr, (candidate, reference) in itertools.product(TAGS, ("EVA", "ETU"), (12, 16, 20, 24), CONTRASTS):
        name = candidate + "_minus_" + reference
        rows = sorted((row for row in contrast_points if
                      (row["tag"], row["channel"], row["snr_db"], row["contrast"]) == (tag, channel, snr, name)),
                      key=lambda row: row["training_seed"])
        require([row["training_seed"] for row in rows] == list(SEEDS), "Missing condition/contrast seed")
        values = [row["delta_ber"] for row in rows]
        contrast_summary.append({"tag": tag, "channel": channel, "snr_db": snr, "contrast": name,
            "candidate": candidate, "reference": reference, "training_seeds": 3,
            "seed1_delta_ber": values[0], "seed2_delta_ber": values[1], "seed3_delta_ber": values[2],
            "mean_delta_ber": float(np.mean(values)), "sample_sd_delta_ber": float(np.std(values, ddof=1))})
    return arm_summary, contrast_summary


def config_tables(seed_arms, seed_contrasts):
    arms, contrasts = [], []
    for tag, arm in itertools.product(TAGS, ARMS):
        rows = [row for row in seed_arms if (row["tag"], row["arm"]) == (tag, arm)]
        values = [row["grid_mean_ber"] for row in rows]
        deltas = [row["delta_from_a0_ber"] for row in rows]
        relative_values = [row["relative_change_from_a0_percent"] for row in rows]
        baseline = float(np.mean([row["a0_grid_mean_ber"] for row in rows]))
        arms.append({"tag": tag, "arm": arm, "training_seeds": 3,
            "seed1_grid_mean_ber": values[0], "seed2_grid_mean_ber": values[1], "seed3_grid_mean_ber": values[2],
            "mean_grid_ber": float(np.mean(values)), "sample_sd_grid_ber": float(np.std(values, ddof=1)),
            "mean_delta_from_a0_ber": float(np.mean(deltas)), "sample_sd_delta_from_a0_ber": float(np.std(deltas, ddof=1)),
            "relative_change_of_means_from_a0_percent": relative(float(np.mean(deltas)), baseline),
            "mean_seed_relative_change_from_a0_percent": None if any(value is None for value in relative_values) else float(np.mean(relative_values)),
            "sample_sd_seed_relative_change_from_a0_percent": None if any(value is None for value in relative_values) else float(np.std(relative_values, ddof=1)),
            "all_three_seed_grid_means_improve_over_a0": all(value < 0 for value in deltas),
            "total_error_bits": sum(row["error_bits"] for row in rows)})
    for tag, (candidate, reference) in itertools.product(TAGS, CONTRASTS):
        name = candidate + "_minus_" + reference
        rows = [row for row in seed_contrasts if (row["tag"], row["contrast"]) == (tag, name)]
        values = [row["delta_ber"] for row in rows]
        relative_values = [row["relative_change_percent"] for row in rows]
        contrasts.append({"tag": tag, "contrast": name, "candidate": candidate, "reference": reference,
            "training_seeds": 3, "seed1_delta_ber": values[0], "seed2_delta_ber": values[1], "seed3_delta_ber": values[2],
            "mean_delta_ber": float(np.mean(values)), "sample_sd_delta_ber": float(np.std(values, ddof=1)),
            "candidate_config_mean_ber": float(np.mean([row["candidate_grid_mean_ber"] for row in rows])),
            "reference_config_mean_ber": float(np.mean([row["reference_grid_mean_ber"] for row in rows])),
            "relative_change_of_means_percent": relative(float(np.mean(values)), float(np.mean([row["reference_grid_mean_ber"] for row in rows]))),
            "mean_seed_relative_change_percent": None if any(value is None for value in relative_values) else float(np.mean(relative_values)),
            "sample_sd_seed_relative_change_percent": None if any(value is None for value in relative_values) else float(np.std(relative_values, ddof=1)),
            "improved_seeds": sum(value < 0 for value in values), "worse_seeds": sum(value > 0 for value in values),
            "exact_tie_seeds": sum(value == 0 for value in values)})
    return arms, contrasts


def training_tables(training_root, protocol_sha):
    costs, curves, ranges, bindings = [], [], [], []
    for tag, seed, arm in itertools.product(TAGS, SEEDS, ("A1", "A1F")):
        directory = training_root / key(tag, seed) / arm
        path = directory / "manifest.json"
        manifest = read_json(path)
        require((manifest["status"], manifest["stage"], manifest["step"], manifest["training_slots"], manifest["arm"]) ==
                ("complete", "train", 3000, 1536000, arm), "Incomplete continuation task")
        require(manifest["protocol_sha256"] == protocol_sha and manifest["selection"] == "final", "Training selection/protocol differs")
        final_path = Path(manifest["final_checkpoint"]["path"])
        require(final_path.resolve() == (directory / "final.pt").resolve() and sha(final_path) == manifest["final_checkpoint"]["sha256"], "Final checkpoint changed")
        elapsed = manifest["elapsed_seconds"]
        require(elapsed > 0 and math.isfinite(elapsed), "Invalid training wall time")
        costs.append({"tag": tag, "training_seed": seed, "arm": arm,
            "observer_mode": manifest["observer_mode"], "updates": 3000, "batch": 512,
            "training_slots": manifest["training_slots"], "wall_seconds": elapsed,
            "optimization_seconds": manifest["optimization_seconds"],
            "non_optimization_overhead_seconds": elapsed - manifest["optimization_seconds"],
            "seconds_per_update": elapsed / 3000, "slots_per_second": manifest["training_slots"] / elapsed,
            "peak_process_memory_bytes": manifest["peak_process_memory_bytes"],
            "training_stimulus_chain_sha256": manifest["training_stimulus_chain_sha256"],
            "final_checkpoint_sha256": manifest["final_checkpoint"]["sha256"],
            "final_model_state_sha256": manifest["model_contract"]["model_state_sha256"]})
        log = directory / "training.jsonl"
        with log.open(encoding="utf-8") as stream:
            records = [json.loads(line) for line in stream]
        require(records and records[-1]["step"] == 3000, "Missing final training log point")
        for row in records:
            curves.append({"tag": tag, "training_seed": seed, "arm": arm, "step": row["step"],
                "loss": row["loss"], "train_batch_ber": row["train_batch_ber"], "lr": row["lr"],
                "elapsed_seconds": row["elapsed_seconds"], "training_slots": row["training_slots"],
                "peak_process_memory_bytes": row["peak_process_memory_bytes"]})
        for module, initial in manifest["initial_activation_ranges"].items():
            final = manifest["final_activation_ranges"][module]
            require(final["amax"] >= initial["amax"], "Running activation maximum decreased")
            if arm == "A1F":
                require(initial == final, "Frozen-range control changed its calibration")
            ranges.append({"tag": tag, "training_seed": seed, "arm": arm, "module": module,
                "initial_amax": initial["amax"], "final_amax": final["amax"],
                "amax_change": final["amax"] - initial["amax"], "bits": final["bits"],
                "ratio_final_to_initial": None if initial["amax"] == 0 else final["amax"] / initial["amax"]})
        bindings.extend({"path": str(p), "sha256": sha(p)} for p in (path, log, final_path))
    return costs, curves, ranges, bindings


def threshold_table(lookup):
    output = []
    for tag, seed, channel, arm in itertools.product(TAGS, SEEDS, ("EVA", "ETU"), ARMS):
        snrs = (12, 16, 20, 24)
        values = [lookup[(tag, seed, channel, snr)]["arms"][arm]["ber"] for snr in snrs]
        record = {"tag": tag, "training_seed": seed, "channel": channel, "arm": arm}
        record.update(observed_threshold(snrs, values))
        for snr, value in zip(snrs, values):
            record[f"raw_ber_{snr}db"] = value
        output.append(record)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--protocol-sha256", required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--training-root", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    protocol_path, run, training_root, audit_path = [path.resolve(strict=True) for path in
        (args.protocol, args.run, args.training_root, args.audit)]
    protocol_sha = sha(protocol_path)
    require(protocol_sha == args.protocol_sha256, "Locked protocol digest differs")
    protocol = read_json(protocol_path)
    manifest, audit = checked_inputs(protocol, protocol_sha, run, audit_path)
    arm_points, contrast_points, lookup, condition_bindings = condition_tables(run, manifest, protocol_sha)
    condition_arms, condition_contrasts = condition_seed_tables(arm_points, contrast_points)
    seed_arms, seed_contrasts = seed_tables(lookup)
    config_arms, config_contrasts = config_tables(seed_arms, seed_contrasts)
    costs, curves, ranges, training_bindings = training_tables(training_root, protocol_sha)
    for arm in ARMS:
        require(sum(row["error_bits"] for row in arm_points if row["arm"] == arm) == audit["error_bits_recomputed"][arm],
                "Summary counts differ from independent audit: " + arm)
    for row in config_contrasts:
        matching = [item for item in audit["all_raw_seed_grid_contrasts"] if
                    (item["tag"], item["contrast"]) == (row["tag"], row["contrast"])]
        require(len(matching) == 1, "Audit seed-grid coverage differs")
        audited = matching[0]
        require(abs(audited["mean_delta"] - row["mean_delta_ber"]) < 1e-15 and
                abs(audited["sample_sd"] - row["sample_sd_delta_ber"]) < 1e-15,
                "Summary seed-grid mean/SD differs from independent audit")
        require(all(abs(a - row[f"seed{seed}_delta_ber"]) < 1e-15
                    for a, seed in zip(audited["raw_seed_deltas"], SEEDS)), "Raw paired seed values differ from audit")
    for row in costs:
        matching = [item for item in audit["training_runs"] if
                    (item["tag"], item["training_seed"], item["arm"]) == (row["tag"], row["training_seed"], row["arm"])]
        require(len(matching) == 1 and matching[0]["checkpoint_sha256"] == row["final_checkpoint_sha256"],
                "Audited training checkpoint differs from summary source")
    thresholds = threshold_table(lookup)
    output = args.output.resolve()
    root = Path(protocol["output_root"]).resolve()
    require(root in output.parents and output != root, "Use a new analysis subdirectory within the locked experiment")
    require(not output.exists(), "Preserve existing summary; choose a new output directory")
    output.mkdir(parents=True, exist_ok=False)
    tables = {"all_points.csv": arm_points, "all_contrasts.csv": contrast_points,
        "condition_arm_seed_summary.csv": condition_arms, "condition_contrast_seed_summary.csv": condition_contrasts,
        "seed_grid_arms.csv": seed_arms, "seed_grid_contrasts.csv": seed_contrasts,
        "config_arm_summary.csv": config_arms, "config_contrast_summary.csv": config_contrasts,
        "training_costs.csv": costs, "training_curves.csv": curves,
        "activation_range_changes.csv": ranges, "observed_thresholds.csv": thresholds}
    for filename, rows in tables.items():
        write_csv(output / filename, rows)
    summary = {"status": "complete", "protocol_sha256": protocol_sha, "conditions": 72,
        "unique_slots": 144000, "unique_bits": 76800000, "training_runs": 18,
        "training_updates_total": sum(row["updates"] for row in costs),
        "training_slots_total": sum(row["training_slots"] for row in costs),
        "training_wall_seconds_sum": sum(row["wall_seconds"] for row in costs),
        "training_time_measurement": "Runner elapsed time through the last update; includes batch generation, intermediate logging/checkpoint overhead; excludes final serialization and process startup",
        "primary_contrast": "A1_minus_A2", "config_arm_summary": config_arms,
        "config_contrast_summary": config_contrasts, "all_seed_arms": seed_arms,
        "all_seed_contrasts": seed_contrasts,
        "uncertainty": "Mean and sample SD across three selected-checkpoint/cohort seeds; not CI or a hypothesis test",
        "conditional_intervals": "Condition-level paired-slot percentile intervals; no training uncertainty or multiplicity adjustment",
        "relative_definition": "100*(candidate grid mean-reference grid mean)/reference grid mean; negative means reduction",
        "threshold_scope": "Secondary observed 12/16/20/24 dB curves only; explicit censoring and no extrapolation",
        "threshold_status_counts": {status: sum(row["status"] == status for row in thresholds)
                                    for status in sorted({row["status"] for row in thresholds})},
        "statistics_performed": "Descriptive summaries only; no significance or equivalence tests",
        "scope": "Software continuation and frozen inference comparison only; original hardware evidence does not validate these new checkpoints"}
    write_json(output / "summary.json", summary)
    bindings = [{"path": str(path), "sha256": sha(path)} for path in
                (protocol_path, run / "manifest.json", audit_path)] + condition_bindings + training_bindings
    audited_bindings = {Path(row["path"]).resolve(): row["sha256"] for row in audit["input_bindings"]}
    for binding in condition_bindings + training_bindings:
        require(audited_bindings.get(Path(binding["path"]).resolve()) == binding["sha256"],
                "Summary artifact was not bound by independent audit: " + binding["path"])
    for binding in audit["input_bindings"]:
        require(sha(binding["path"]) == binding["sha256"], "Audited input changed during summarization")
    for binding in bindings:
        require(sha(binding["path"]) == binding["sha256"], "Input changed during summarization")
    provenance = {"status": "complete", "protocol_path": str(protocol_path), "protocol_sha256": protocol_sha,
        "evaluation_manifest_sha256": sha(run / "manifest.json"), "independent_audit_path": str(audit_path),
        "independent_audit_sha256": sha(audit_path), "independent_audit_status": audit["status"],
        "summarizer_sha256": sha(Path(__file__)), "input_bindings": bindings,
        "outputs": {path.name: {"sha256": sha(path), "rows": len(tables[path.name]) if path.name in tables else None}
                    for path in output.iterdir() if path.is_file()},
        "environment": {"python": platform.python_version(), "numpy": np.__version__},
        "source_integrity": "Every condition retained; raw bit/slot conservation rechecked; independent audit required",
        "figure_data_scope": "These are real final-test and training artifacts; no mock observations"}
    write_json(output / "manifest.json", provenance)
    print(json.dumps({"status": "complete", "output": str(output), "conditions": 72,
                      "training_runs": 18, "primary_contrast": "A1_minus_A2"}), flush=True)


if __name__ == "__main__":
    main()
