"""Generate SI tables from the audited, complete continuation CSVs.

This exports recorded numbers only. It does not train, evaluate, interpolate
new test points, alter scientific values or decide which result to emphasize.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path


LABELS = {"b1_longcp": "BPSK/CP16", "b2_longcp": "QPSK/CP16", "b2_shortcp": "QPSK/CP4"}
ARMS = ("A0", "A1", "A2", "A1F", "A1A12")
CONTRASTS = ("A1_minus_A2", "A1_minus_A0", "A2_minus_A0", "A1F_minus_A0", "A1F_minus_A1", "A1A12_minus_A1")
ROW_END = " " + chr(92) * 2


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def csv_rows(path):
    with Path(path).open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def table(caption, label, columns, header, rows):
    return "\n".join([r"\begin{table}[!htbp]", r"\centering", r"\small",
        r"\setlength{\tabcolsep}{4pt}", "\\caption{" + caption + "}",
        "\\label{" + label + "}", "\\begin{tabular}{" + columns + "}",
        r"\toprule", header + ROW_END, r"\midrule", *[row + ROW_END for row in rows],
        r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])


def number(value, scale=1000):
    return f"{float(value) * scale:.3f}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary = args.summary.resolve(strict=True)
    manifest = load(summary / "manifest.json")
    report = load(summary / "summary.json")
    if manifest["status"] != "complete" or manifest["independent_audit_status"] != "pass" or report["conditions"] != 72:
        raise ValueError("Completed independent audit and all planned conditions required")
    for name, binding in manifest["outputs"].items():
        if sha(summary / name) != binding["sha256"]:
            raise ValueError("Source-data digest changed: " + name)
    output = args.output.resolve()
    if output.exists():
        raise ValueError("Use a new table-export directory")
    output.mkdir(parents=True)
    artifacts = {}

    def save(name, text):
        path = output / name
        with path.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(text)
        artifacts[name] = sha(path)

    seed_arms = csv_rows(summary / "seed_grid_arms.csv")
    config_arms = csv_rows(summary / "config_arm_summary.csv")
    seed_contrasts = csv_rows(summary / "seed_grid_contrasts.csv")
    config_contrasts = csv_rows(summary / "config_contrast_summary.csv")
    if (len(seed_arms), len(config_arms), len(seed_contrasts), len(config_contrasts)) != (45, 15, 54, 18):
        raise ValueError("Incomplete grid-table inventory")
    rows = []
    for tag in LABELS:
        for arm in ARMS:
            raw = sorted([row for row in seed_arms if row["tag"] == tag and row["arm"] == arm],
                         key=lambda row: int(row["training_seed"]))
            aggregate, = [row for row in config_arms if row["tag"] == tag and row["arm"] == arm]
            if [int(row["training_seed"]) for row in raw] != [1, 2, 3]:
                raise ValueError("Raw-seed inventory changed")
            cells = [LABELS[tag], arm] + [number(row["grid_mean_ber"]) for row in raw]
            cells += ["$" + number(aggregate["mean_grid_ber"]) + r"\pm" + number(aggregate["sample_sd_grid_ber"]) + "$"]
            rows.append(" & ".join(cells))
    save("continued_qat_grid_arms.tex", table(
        "All five arms on the new continued-QAT final-test cohort. Equal-weight eight-condition grid BER in units of $10^{-3}$. Each seed averages EVA/ETU and 12/16/20/24 dB, with 2,000 common slots per condition. Mean and sample SD use the three raw starting-checkpoint/cohort seeds, not three new from-scratch training chains or a confidence interval.",
        "tab:continuedarms", "llrrrr", "Configuration & Arm & Seed 1 & Seed 2 & Seed 3 & Mean $\\pm$ SD", rows))
    rows = []
    for tag in LABELS:
        for contrast in CONTRASTS:
            raw = sorted([row for row in seed_contrasts if row["tag"] == tag and row["contrast"] == contrast],
                         key=lambda row: int(row["training_seed"]))
            aggregate, = [row for row in config_contrasts if row["tag"] == tag and row["contrast"] == contrast]
            if [int(row["training_seed"]) for row in raw] != [1, 2, 3]:
                raise ValueError("Raw-seed contrast inventory changed")
            candidate, reference = contrast.split("_minus_")
            name = candidate + "--" + reference
            if contrast == "A1_minus_A2":
                name = r"\textbf{" + name + "}"
            cells = [LABELS[tag], name] + [number(row["delta_ber"]) for row in raw]
            cells += ["$" + number(aggregate["mean_delta_ber"]) + r"\pm" + number(aggregate["sample_sd_delta_ber"]) + "$"]
            rows.append(" & ".join(cells))
    save("continued_qat_grid_contrasts.tex", table(
        "All six predeclared contrasts on the same final cohort, in $10^{-3}$ BER. Negative values mean fewer errors for the first arm. Bold identifies the locked primary A1--A2 contrast regardless of its direction, not a selected winner. Every configuration retains all three seeds; sample SD is distinct from the condition-level paired-slot intervals in the source data.",
        "tab:continuedcontrasts", "llrrrr", "Configuration & Contrast & Seed 1 & Seed 2 & Seed 3 & Mean $\\pm$ SD", rows))
    costs = csv_rows(summary / "training_costs.csv")
    if len(costs) != 18:
        raise ValueError("Incomplete training-cost inventory")
    rows = []
    for row in costs:
        rows.append(" & ".join([LABELS[row["tag"]], row["training_seed"], row["arm"],
                    f"{float(row['wall_seconds']):.1f}", f"{float(row['seconds_per_update']):.4f}",
                    f"{int(row['peak_process_memory_bytes']) / (1024 ** 2):.1f}"]))
    save("continued_qat_training_costs.tex", table(
        "All 18 fixed-final continuation runs: 3,000 updates, batch 512, and 1,536,000 model-slot exposures each. Time follows the retained trajectory through the last update, including stimulus generation, logging and intermediate checkpoints, but excluding startup and final serialization. For three resumed runs it combines the saved-prefix and resumed times; discarded unsaved-tail work and the interruption gap are not included. The retained logs additionally document 1,137 recomputed updates. Three processes run concurrently, each with two PyTorch threads. These are not equal-compute controls, energy measurements, or FPGA execution costs. Peak memory is per process, in MiB.",
        "tab:continuedcost", "llrrrr", "Configuration & Seed & Arm & Time (s) & s/update & Peak MiB", rows))
    thresholds = csv_rows(summary / "observed_thresholds.csv")
    if len(thresholds) != 90:
        raise ValueError("Incomplete observed-threshold inventory")
    rows = []
    for tag in LABELS:
        for arm in ARMS:
            cells = [LABELS[tag], arm]
            for channel in ("EVA", "ETU"):
                for seed in (1, 2, 3):
                    row, = [item for item in thresholds if item["tag"] == tag and item["arm"] == arm
                            and item["channel"] == channel and int(item["training_seed"]) == seed]
                    status = row["status"]
                    if status == "observed_crossing":
                        cells.append(f"{float(row['threshold_db']):.3f}")
                    elif status == "left_censored":
                        cells.append(r"$\leq12$")
                    elif status == "right_censored":
                        cells.append(r"$>24$")
                    else:
                        cells.append("Bracket only")
            rows.append(" & ".join(cells))
    save("continued_qat_observed_thresholds.tex", table(
        "Secondary 2\\%-BER thresholds on only the new 12/16/20/24-dB grid. Columns are seeds 1--3 for each channel. Observed crossings use the running-minimum envelope and SNR--log-BER interpolation. Censored cells are bounds, not numerical threshold estimates; no historical 0--28-dB points are inserted. These are software results for the continuation cohort, not the historical benchmark thresholds.",
        "tab:continuedthresholds", "llrrrrrr", "Configuration & Arm & EVA S1 & EVA S2 & EVA S3 & ETU S1 & ETU S2 & ETU S3", rows))
    ledger = {"status": "complete", "protocol_sha256": report["protocol_sha256"],
        "summary_manifest_sha256": sha(summary / "manifest.json"), "source_data": str(summary),
        "source_csv_sha256": {name: sha(summary / name) for name in ("seed_grid_arms.csv", "seed_grid_contrasts.csv",
            "config_arm_summary.csv", "config_contrast_summary.csv", "training_costs.csv", "observed_thresholds.csv")},
        "script_sha256": sha(__file__), "outputs": artifacts, "rounding": "3 decimals for 0.001 BER and dB; runtime 1 decimal; only display rounding",
        "no_test_generation_or_model_evaluation": True}
    with (output / "manifest.json").open("x", encoding="utf-8") as stream:
        json.dump(ledger, stream, indent=2)
    print(json.dumps({"status": "complete", "output": str(output), "tables": list(artifacts)}))


if __name__ == "__main__":
    main()
