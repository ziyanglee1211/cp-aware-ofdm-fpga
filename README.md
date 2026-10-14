# CP-aware neural OFDM receiver: reproducibility package

Local release candidate `v1.0.0`, prepared 14 October 2026, for the author-edited manuscript **Structure- and Sensitivity-Guided FPGA Mapping of a CP-Aware Complex Neural OFDM Receiver** (target: Integration, the VLSI Journal). This is not a claim of acceptance or publication. No article DOI or public repository URL has been assigned in this local package.

Authors, in manuscript order: Ziyang Li, Youyao Liu, Lele Wang. The manuscript's corresponding author is Youyao Liu. The manuscript/SI SHA256 bindings are recorded in `provenance/manuscript_binding.json`; the manuscripts themselves and private submission documents are not distributed here.

## What is included

- Original corrected-PDP research code: OFDM channels, neural models, conventional baselines, training/evaluation, component controls, precision interventions and continued-QAT controls.
- Machine-readable plotted values, all continued-QAT summary tables, negative outcomes and seed reversals, original source hashes and fixed reference PDFs for main Figures 1–10.
- MATLAB source inputs and editable native FIG objects for Figures 5–10.
- Authors' local-A12 C/HLS source, coefficient/stimulus headers, wrapper, constraints, ROMs and archived machine-readable implementation evidence. Proprietary generated IP and tool installations are excluded.

The **complete Zenodo archive** additionally contains original checkpoints, formal per-slot NPZ outcomes, component seed records and the selected final training records. The source tree in that archive is the same as this GitHub tree. Obtain that archive and extract it into a new directory to run all recorded-data audits and checkpoint tests. Do not infer completeness from cloning GitHub alone.

## Quick start: integrity and recorded-data audit

The full archive needs Python and NumPy for the numerical audit, not a GPU:

```bash
python tools/verify_release.py
python tools/audit_results.py
```

`verify_release.py` uses the Python standard library and checks the file manifest. `audit_results.py` uses NumPy, re-counts saved slot outcomes, reconstructs continued-QAT configuration means/sample SD, checks the primary contrast and archived hardware counts, and prints JSON to stdout. Neither command retrains models, writes into research evidence, programs a board, or creates a new performance measurement. On the lightweight GitHub tree, missing large data are explicitly reported as missing rather than silently treated as passed.

For checkpoint development tests, use a compatible PyTorch environment and the full archive:

```bash
cd research/dccn
python -m unittest test_paired_research_stats test_back_precision_followup test_continued_qat_control
```

These are contract/development tests, not a rerun of the full 3,000-update study. See `docs/REPRODUCTION.md` before launching a historical experiment script.

## Figure exports

In MATLAB R2021b, from the repository root:

```matlab
addpath('tools');
export_native_figures(fullfile(pwd,'generated_figures'));
```

This exports the retained native numerical objects for Figures 5–10 into a **new** directory. It does not rerun experiments. Re-exported font encoding/metadata can differ from the frozen submission PDFs; the latter are retained byte-for-byte in `figures/reference/`. Source-data reconstruction and object re-export are different verification levels. Figures 1–4 are explanatory schematics, not measured outputs.

## Scientific boundaries

- PDP powers are converted from dB to linear power and normalized. Legacy direct-dB normalization is not the manuscript's formal channel route.
- Original A12 is a frozen, output-only back-mixing activation intervention. Do not assign original hardware metrics to continued-QAT weights or QPSK configurations.
- Training-seed summaries use sample SD; paired slot intervals are conditional on a checkpoint, not uncertainty across independently trained models.
- The mapped example's 58,115-cycle number is an HLS single-call estimate, not measured sustained interface throughput. Its recorded board check covers two ROM-fed slots / 640 unique decisions, not live RF or a universal bit-exact theorem.
- Negative, adverse and reversed outcomes are retained. Original historical protocol files remain unchanged; workstation paths inside them are historical locators, not instructions to write to those paths.

## Licensing and provenance

Original code/hardware source is provided under MIT; original research data and model/figure artifacts are under CC BY 4.0 as scoped in `LICENSE-DATA.md`. Third-party PDP matrices retain their upstream MIT notice. Read `THIRD_PARTY_NOTICES.md` and `docs/PROVENANCE.md`. Public permission does not transfer ownership of third-party material.

Do not cite a fabricated DOI. Once a real Zenodo record exists, add its URL/DOI and the GitHub URL in a new documented metadata change, then reference the precise archived version used for the manuscript.
