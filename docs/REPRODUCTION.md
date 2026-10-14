# Reproduction levels and evidence map

## Levels

1. **Integrity:** `python tools/verify_release.py` verifies the complete payload manifest. This is not scientific replication.
2. **Recorded-data reanalysis:** `python tools/audit_results.py` reconstructs error counts and continued-QAT grid summaries from retained outcomes. This does not regenerate traffic or retrain models.
3. **Drawing/object export:** native FIG exports reproduce stored drawing objects. JSON/MATLAB data plotting sources are included for Figures 5–8 and 10; the exact submission PDFs remain reference outputs. Re-export is not promised to be byte-identical across font/tool versions.
4. **Inference/training rerun:** research code and checkpoints are supplied with recorded settings. Historical Windows paths and protected-output protocols must be adapted in a separate rerun namespace. The full 54,000 retained continuation updates were not rerun during packaging.
5. **Hardware rebuild:** original C/HLS source, wrapper, ROMs, constraints and Tcl are supplied. Obtain the licensed 2022.2 tools and regenerate the HLS IP yourself. Hardware scripts have historical absolute paths and reset/new-project guards; do not run them blindly. No vendor-generated IP, bitstream or DLL is redistributed, and no hardware was programmed during packaging.

## Main figures

| Figure | Included recorded inputs | Role |
| --- | --- | --- |
| 1–4 | Fixed reference PDFs; OFDM configuration and receiver/C source | Explanatory design/quantization/architecture, not a new measured dataset |
| 5 | `figures/matlab/fig_ber_paired_b1_longcp_source.json`, paired and Wiener seed results under `research/dccn/results` | Corrected-PDP nine-method BER and paired threshold comparison |
| 6 | Component protocol/summary; complete seed records and final checkpoints in full archive | Seven-arm common warm-start budget intervention |
| 7 | All 22 precision variants in summary and formal screening NPZ records | Frozen-checkpoint sensitivity localization |
| 8 | Confirmation protocol/decision, all confirmation outcomes and native-C transfer records | Selected local-A12 confirmation and checkpoint-specific transfer |
| 9 | All 12 continued-QAT CSV tables, 72 five-arm conditions, selected 18 final training records | Fixed-budget adaptation challenge, including unfavorable results |
| 10 | Recorded HLS/route counts and timing in local-A12 closure JSON, plus plotted source JSON | Matched scheduling estimates and mapped resource headroom |

## Original software entry points

Under `research/dccn`: `train.py`, `struct_train.py`, `struct_qat_train.py`; `eval_paired_b1_longcp_pdpfix.py`, `eval_pdp_wiener_b1_longcp.py`; `run_component_ablation_pdpfix.py`, `run_precision_localization_pdpfix.py`, `run_back_precision_followup.py`; and `run_continued_qat_control.py`. Inspect their arguments and recorded protocols before selecting an experiment. Do not run every historical script sequentially as a reproduction recipe.

The continued-QAT `summary/manifest.json` selects the actual final checkpoints, including recovery-directory finals where applicable. Intermediate, partial and pilot checkpoints were not substituted. Source records and original checkpoint hashes remain available. New runs may differ with tool versions, RNG behavior and platform; distinguish a rerun from reproduction of the archived counts.

## Hardware boundary

The native bridge and C implementation can be rebuilt using an appropriate Windows C++ compiler. The recorded flags were `-O2 -std=c++11 -ffp-contract=off -shared -static-libgcc -static-libstdc++`, without fast-math. Rebuild both rev-T and local-A12 libraries if rerunning transfer evaluation; the archived evidence has 61 and 42 C/PyTorch decision differences respectively over the transfer campaign. Do not claim universal C/PyTorch identity.

Historical route/self-test evidence is limited to the original mapped BPSK/CP16 seed-1 instance. Continued-QAT checkpoints, other seeds and QPSK have software evidence only. The archive intentionally does not imply measured sustained I/O throughput, live-RF reception or measured board power.
