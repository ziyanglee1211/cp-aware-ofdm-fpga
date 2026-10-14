"""Freeze the A12-output candidate before evaluating a second fresh cohort.

Uses the same verified runner with a reduced, explicit variant set. The
selection protocol binds this entry point and the preceding nomination data.
"""
import json
import sys
from pathlib import Path

import run_back_precision_followup as followup
from run_precision_localization_pdpfix import file_hash, save_json


def main():
    base = followup.DEFAULT_OUT
    nomination = base / "independent_2000"
    audit = json.loads((base / "independent_2000_audit/audit.json").read_text(encoding="utf-8"))
    assert audit["status"] == "pass"
    original_specs = followup.SPECS
    names = ("baseline_fft", "baseline_matrix", "back_out_a12", "back_out_a16")
    followup.SPECS = {name: original_specs[name] for name in names}
    protocol_path = base / "confirmation_protocol.json"
    if protocol_path.exists():
        raise SystemExit("Confirmation already started; use preserved evidence")
    protocol = {
        "status": "locked_before_confirmation", "cohort": 2026100202,
        "selected_candidate": "back_out_a12",
        "selection_reason": "The nominated back-output boundary benefits all nine config/seed grid means. A12 is the smaller finite-width output-only candidate tested; its screening grid means are close to A16. This does not establish equivalence or hardware fit.",
        "confirmation_comparators": list(names), "variants": followup.SPECS,
        "primary_metric": "candidate-minus-FFT W8A8 equal-weight BER over EVA/ETU x 12/16/20/24 dB, separately for each configuration and seed",
        "decision_rule": "Report every seed and condition; candidate progresses only if all nine seed grid means improve. Do not retune widths, weights, calibration or conditions on confirmation traffic.",
        "slots_per_condition": 2000, "bootstrap_reps": 2000,
        "nomination_manifest_sha256": file_hash(nomination / "manifest.json"),
        "nomination_summary_sha256": file_hash(nomination / "grid_summary.json"),
        "nomination_audit_sha256": file_hash(base / "independent_2000_audit/audit.json"),
        "confirmation_entry_sha256": file_hash(Path(__file__)),
        "runner_sha256": file_hash(Path(followup.__file__)),
        "scope": "Independent simulated traffic, same three pretrained checkpoints; no independent retraining, new channel distribution or hardware validation."}
    save_json(protocol_path, protocol)
    output = base / "confirmation_2000"
    sys.argv = [str(Path(followup.__file__)), "--output", str(output), "--cohort", "2026100202"]
    followup.main()
    screen_hashes = {json.loads(p.read_text(encoding="utf-8"))["stimulus_sha256"]["received"]
                     for p in nomination.glob("*_seed*dB.json")}
    confirm_hashes = {json.loads(p.read_text(encoding="utf-8"))["stimulus_sha256"]["received"]
                      for p in output.glob("*_seed*dB.json")}
    assert len(confirm_hashes) == 72 and screen_hashes.isdisjoint(confirm_hashes)
    summary = json.loads((output / "grid_summary.json").read_text(encoding="utf-8"))
    deltas = {tag: group["back_out_a12"]["raw_seed_deltas"] for tag, group in summary["tags"].items()}
    record = {"status": "complete", "confirmation_protocol_sha256": file_hash(protocol_path),
              "confirmation_manifest_sha256": file_hash(output / "manifest.json"),
              "confirmation_summary_sha256": file_hash(output / "grid_summary.json"),
              "received_cohorts_disjoint_from_nomination": True,
              "selected_candidate": "back_out_a12", "raw_seed_grid_deltas": deltas,
              "all_nine_seed_grid_means_improve": all(x < 0 for values in deltas.values() for x in values),
              "scope": protocol["scope"]}
    save_json(base / "confirmation_decision.json", record)
    print(json.dumps(record, indent=2), flush=True)


if __name__ == "__main__":
    main()
