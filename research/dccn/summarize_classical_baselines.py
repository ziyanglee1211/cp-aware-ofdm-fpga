"""Aggregate formal classical-baseline runs and create a paper-ready BER plot."""

import argparse
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def snr_at_ber(snrs, bers, target):
    curve = np.minimum.accumulate(np.asarray(bers, dtype=float))
    grid = np.asarray(snrs, dtype=float)
    if curve[-1] > target:
        return None
    for i in range(len(curve) - 1):
        if curve[i] > target >= curve[i + 1]:
            lo, hi = np.log10(curve[i]), np.log10(curve[i + 1])
            w = (np.log10(target) - lo) / (hi - lo) if hi != lo else 0.0
            return float(grid[i] + w * (grid[i + 1] - grid[i]))
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pattern", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--figure", required=True)
    ap.add_argument("--target-ber", type=float, default=2e-2)
    args = ap.parse_args()

    paths = sorted(glob.glob(args.pattern))
    if not paths:
        raise FileNotFoundError(args.pattern)
    runs = []
    for path in paths:
        with open(path, encoding="utf-8") as f:
            runs.append(json.load(f))

    snrs = runs[0]["snrs"]
    methods = list(runs[0]["results"])
    summary = {
        "source_files": [os.path.abspath(p) for p in paths],
        "n_eval_seeds": len(runs),
        "system": runs[0]["system"],
        "snrs": snrs,
        "target_ber": args.target_ber,
        "results": {},
    }
    for method in methods:
        curves = np.asarray([r["results"][method] for r in runs], dtype=float)
        crossings = [snr_at_ber(snrs, curve, args.target_ber) for curve in curves]
        valid = np.asarray([v for v in crossings if v is not None])
        summary["results"][method] = {
            "ber_mean": curves.mean(axis=0).tolist(),
            "ber_std": (curves.std(axis=0, ddof=1).tolist()
                        if len(curves) > 1 else [0.0] * len(snrs)),
            "snr_at_target_per_seed": crossings,
            "snr_at_target_mean": float(valid.mean()) if len(valid) else None,
            "snr_at_target_std": (float(valid.std(ddof=1)) if len(valid) > 1
                                  else (0.0 if len(valid) == 1 else None)),
        }

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    styles = {
        "ls_spline": ("s--", "LS-Spline + ZF"),
        "lmmse": ("^-.", "Ideal LMMSE + ZF"),
        "almmse": ("D--", "ALMMSE + ZF"),
        "ls_cp": ("o-", "LS-CP"),
        "perfect": ("x:", "Perfect CSI + ZF"),
    }
    plt.figure(figsize=(7.2, 5.2))
    for method in methods:
        mean = summary["results"][method]["ber_mean"]
        style, label = styles[method]
        plt.semilogy(snrs, mean, style, label=label, linewidth=1.7,
                     markersize=5)
    sys = summary["system"]
    plt.xlabel("Es/N0 (dB)")
    plt.ylabel("BER")
    plt.title(f"Classical baselines: {sys['channel']}, "
              f"{'BPSK' if sys['nbits']==1 else 'QPSK'}, {sys['cp']} CP")
    plt.grid(True, which="both", alpha=0.3)
    plt.ylim(1e-4, 1)
    plt.legend()
    os.makedirs(os.path.dirname(os.path.abspath(args.figure)), exist_ok=True)
    plt.savefig(args.figure, dpi=200, bbox_inches="tight")

    print(f"aggregated {len(runs)} run(s) -> {args.output}")
    for method in methods:
        row = summary["results"][method]
        if row["snr_at_target_mean"] is None:
            crossing = "no crossing"
        else:
            crossing = (f"{row['snr_at_target_mean']:.2f} +/- "
                        f"{row['snr_at_target_std']:.2f} dB")
        print(f"{method:10s}: {crossing}")
    print("saved", args.figure)


if __name__ == "__main__":
    main()
