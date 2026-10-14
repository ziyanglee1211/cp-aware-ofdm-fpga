"""Independently rebuild follow-up summaries from paired per-slot counts."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from paired_research_stats import summarize_seed_deltas
from run_precision_localization_pdpfix import file_hash, save_json

HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=HERE / "experiments/back_precision_followup_20261002/independent_2000")
    args = parser.parse_args()
    folder = args.run.resolve()
    output = folder.parent / (folder.name + "_audit")
    if output.exists():
        raise SystemExit("Use a run whose audit output does not already exist")
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    if manifest["status"] != "complete" or manifest["smoke_not_for_publication"]:
        raise AssertionError("Require completed formal run")
    specifications = manifest["variants_fixed_before_run"]
    settings = manifest["args"]
    conditions = manifest["conditions"]
    expected = len(settings["tags"]) * len(settings["seeds"]) * len(settings["channels"]) * len(settings["snrs"])
    assert len(conditions) == len(set(conditions)) == expected
    for name, digest in manifest["source_hashes"].items():
        assert file_hash(HERE / name) == digest, name
    for row in manifest["checkpoints"].values():
        assert file_hash(row["path"]) == row["sha256"]
    raw = []
    stimuli, namespaces, keys = set(), set(), set()
    baseline_total, bits_total = {}, {}
    for name in conditions:
        point = json.loads((folder / name).read_text(encoding="utf-8"))
        key = (point["tag"], point["training_seed"], point["channel"], point["snr_db"])
        assert key not in keys
        keys.add(key)
        assert point["slots"] == settings["slots"]
        assert point["pdp_mode"] == "3gpp_db_to_linear_power"
        assert point["baseline_max_absolute_difference_all_slots"] == {"fft": 0, "matrix": 0}
        assert set(point["variants"]) == set(specifications)
        stimuli.add(point["stimulus_sha256"]["received"])
        namespaces.add(point["point_seed"])
        slot_path = folder / point["slot_data"]
        assert file_hash(slot_path) == point["slot_data_sha256"]
        with np.load(slot_path, allow_pickle=False) as data:
            reference = data["baseline_slot_errors"]
            assert reference.shape == (settings["slots"],)
            assert np.array_equal(reference, data["baseline_fft__slot_errors"])
            bits = point["bits_per_slot"]
            tag = point["tag"]
            baseline_total[tag] = baseline_total.get(tag, 0) + int(reference.sum())
            bits_total[tag] = bits_total.get(tag, 0) + len(reference) * bits
            for variant in specifications:
                candidate = data[variant + "__slot_errors"]
                corrected, harmed = data[variant + "__corrected"], data[variant + "__harmed"]
                flips = data[variant + "__flips"]
                assert candidate.shape == reference.shape
                assert np.issubdtype(candidate.dtype, np.integer)
                assert np.all((candidate >= 0) & (candidate <= bits))
                assert np.array_equal(candidate - reference, harmed - corrected)
                assert np.array_equal(flips, harmed + corrected)
                delta = float((candidate.astype(float) - reference).mean() / bits)
                summary = point["variants"][variant]
                assert int(candidate.sum()) == summary["candidate_error_bits"]
                assert int(reference.sum()) == summary["reference_error_bits"]
                assert abs(delta - summary["candidate_minus_reference_ber"]) < 1e-15
                assert int(flips.sum()) == summary["changed_decisions"]
                assert int(corrected.sum()) == summary["corrected_decisions"]
                assert int(harmed.sum()) == summary["harmed_decisions"]
                if variant != "baseline_fft":
                    assert summary["bootstrap_reps"] == settings["bootstrap"]
                    ci = summary["conditional_slot_bootstrap_95pct_ci"]
                    assert len(ci) == 2 and np.isfinite(ci).all() and ci[0] <= ci[1]
                raw.append({"tag": tag, "seed": point["training_seed"], "channel": point["channel"],
                            "snr_db": point["snr_db"], "variant": variant, "delta_ber": delta,
                            "candidate_errors": int(candidate.sum()), "reference_errors": int(reference.sum()),
                            "bits": len(reference) * bits})
    assert len(stimuli) == len(namespaces) == len(conditions)
    summary = json.loads((folder / "grid_summary.json").read_text(encoding="utf-8"))
    grid = []
    for tag in settings["tags"]:
        for variant in specifications:
            deltas = []
            for seed in settings["seeds"]:
                selected = [r for r in raw if r["tag"] == tag and r["seed"] == seed and r["variant"] == variant]
                assert len(selected) == len(settings["channels"]) * len(settings["snrs"])
                deltas.append(float(np.mean([r["delta_ber"] for r in selected])))
            stats = summarize_seed_deltas(deltas)
            recorded = summary["tags"][tag][variant]
            assert np.allclose(stats["raw_seed_deltas"], recorded["raw_seed_deltas"], rtol=0, atol=1e-15)
            assert abs(stats["mean_delta"] - recorded["mean_delta"]) < 1e-15
            assert abs(stats["sample_sd"] - recorded["sample_sd"]) < 1e-15
            baseline_ber = baseline_total[tag] / bits_total[tag]
            grid.append({"tag": tag, "variant": variant, "mean_delta_ber": stats["mean_delta"],
                         "sample_sd": stats["sample_sd"], "raw_seed_deltas": json.dumps(deltas),
                         "baseline_grid_ber": baseline_ber,
                         "candidate_grid_ber": baseline_ber + stats["mean_delta"],
                         "relative_change_percent": 100 * stats["mean_delta"] / baseline_ber})
    output.mkdir()
    for name, rows in (("condition_contrasts.csv", raw), ("grid_summary.csv", grid)):
        with (output / name).open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    result = {"status": "pass", "conditions": len(conditions), "variants": len(specifications),
              "unique_slots": sum(settings["slots"] for _ in conditions), "unique_bits": sum(bits_total.values()),
              "all_slot_baseline_original_agreement": "max absolute difference 0 for both matrix and FFT on every simulated slot",
              "paired_counts_and_grid_summaries_recomputed": True,
              "source_and_original_checkpoint_hashes_unchanged": True,
              "all_received_hashes_and_rng_namespaces_unique": True,
              "manifest_sha256": file_hash(folder / "manifest.json"),
              "summary_sha256": file_hash(folder / "grid_summary.json"),
              "auditor_sha256": file_hash(Path(__file__)),
              "scope": "Descriptive paired fixed-checkpoint follow-up; finite-bit candidates are untrained and unsynthesized."}
    save_json(output / "audit.json", result)
    lines = ["# 后混合精度分解：独立样本复核", "", f"完成 {len(conditions)} 条件 × {len(specifications)} 组；每条件 {settings['slots']} 个共同 coherence slots。",
             "", "以下为两信道 × 四个 SNR 等权网格内 ΔBER，单位 10^-3，负值表示错误减少；均值 ± 三个训练检查点/测试 cohort 对比的样本 SD。它不是 dB 增益、置信区间或独立训练的新模型。", "",
             "| 干预 | BPSK 长 CP | QPSK 长 CP | QPSK 短 CP |", "|---|---:|---:|---:|"]
    for variant in specifications:
        cells = []
        for tag in settings["tags"]:
            r = next(r for r in grid if r["tag"] == tag and r["variant"] == variant)
            cells.append(f"{r['mean_delta_ber']*1000:+.3f} ± {r['sample_sd']*1000:.3f}")
        lines.append("| " + variant + " | " + " | ".join(cells) + " |")
    lines += ["", f"本次预定的 {len(specifications)} 组均保留，包括零效应和变差结果。每一非基线对比提供 {settings['bootstrap']:,} 次时隙配对 bootstrap 条件区间；这些区间不包含训练随机性，也未进行多重比较显著性校正。",
              "", f"全部 {result['unique_slots']:,} 个时隙上的诊断基线分别与共享 matrix/FFT 模型逐值相等；matrix 与 FFT 彼此仍可能出现判决翻转。",
              "", "固定检查点、量化范围及软件算子进行推理干预。没有重训、重新校准、生成 C/RTL 或更新板卡镜像。当前硬件仍为 BPSK 长 CP 的原 W8A8 rev-T。",
              "", "逐条件结果见 condition_contrasts.csv；三 seed 网格均值和原始 seed 差值见 grid_summary.csv；计数、哈希与口径复核见 audit.json。"]
    (output / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    for r in grid:
        if r["variant"] in ("float_in", "float_out", "float_w", "back_out_a12", "back_out_a16", "back_out_ifft_a16"):
            print(r["tag"], r["variant"], f"{r['mean_delta_ber']*1000:+.4f} +/- {r['sample_sd']*1000:.4f} x1e-3", r["raw_seed_deltas"])


if __name__ == "__main__":
    main()
