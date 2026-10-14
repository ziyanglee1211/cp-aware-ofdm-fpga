"""Formal paired evaluation of LMMSE, ALMMSE and LS-CP baselines.

The random seed convention matches E8: evaluation seed 20260904 + seed_id,
reset for every SNR point.  The output can therefore be compared directly with
``structq_*_e8_seed*.json`` without changing the channel/noise protocol.

Example
-------
python eval_classical_baselines.py --nbits 1 --cp short --channel ETU \
  --seed-id 1 --slots 2000
"""

import argparse
import json
import os
import time

import numpy as np
import torch

from benchmarks import ConventionalReceiver, ofdm_frontend
from config import OfdmConfig
from ofdm import ChannelTorch, OfdmTxTorch
from train import apply_fading_awgn


DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")


def count_errors(pred, target):
    return int((pred != target).sum().item())


def evaluate(cfg, channel, snrs, seed_id, slots, cp_batch):
    tx = OfdmTxTorch(cfg, DEV)
    ch = ChannelTorch(cfg, DEV, model=channel)
    conv = ConventionalReceiver(cfg, DEV)
    methods = ("ls_spline", "lmmse", "almmse", "ls_cp", "perfect")
    curves = {name: [] for name in methods}
    counts = {name: [] for name in methods}
    eval_seed = 20260904 + seed_id

    for snr in snrs:
        started = time.time()
        torch.manual_seed(eval_seed + int(snr * 10))
        x_cp, _, labels_pm1 = tx.transmit(slots)
        snr_v = torch.full((slots,), float(snr), device=DEV)
        y_cp, taps = apply_fading_awgn(ch, x_cp, snr_v)
        Y = ofdm_frontend(y_cp, cfg)
        H_true = torch.fft.fft(taps, n=cfg.K, dim=1)
        H_true = H_true.unsqueeze(1).expand(-1, cfg.N_SYM, -1)
        H_ls = conv.ls_spline(Y)

        with torch.no_grad():
            predictions = {
                "ls_spline": conv.detect(Y, H_ls),
                "lmmse": conv.detect(Y, conv.ideal_lmmse(
                    H_ls, H_true, float(snr))),
                "almmse": conv.detect(Y, conv.approx_lmmse(
                    H_ls, float(snr))),
                "perfect": conv.detect(Y, H_true.contiguous()),
            }

            cp_parts = []
            for start in range(0, slots, cp_batch):
                stop = min(start + cp_batch, slots)
                X_cp = conv.ls_cp_equalize(
                    Y[start:stop], H_ls[start:stop], y_cp[start:stop])
                cp_parts.append(conv.detect_equalized(X_cp).cpu())
            predictions["ls_cp"] = torch.cat(cp_parts, dim=0).to(DEV)

            nbits = labels_pm1.numel()
            for name in methods:
                errors = count_errors(predictions[name], labels_pm1)
                counts[name].append(errors)
                curves[name].append(errors / nbits)

        elapsed = time.time() - started
        text = "  ".join(f"{m}={curves[m][-1]:.4e}" for m in methods)
        print(f"{channel} {snr:4.0f} dB  {text}  "
              f"({slots} slots, {elapsed:.1f}s)", flush=True)

    return {
        "protocol": {
            "eval_seed": eval_seed,
            "seed_rule": "20260904 + seed_id; reset with + int(snr_db*10)",
            "slots_per_snr": slots,
            "paired_channel_noise": True,
            "pdp_mode": ch.pdp_mode,
            "pdp_db": ch.pdp_db.detach().cpu().tolist(),
            "pdp_linear_normalized": ch.pdp.detach().cpu().tolist(),
            "lmmse": "ideal true-channel rank-one covariance; reference eq_idx=4",
            "almmse": "LS-estimated rank-one covariance; reference eq_idx=7",
            "ls_cp": "CP-enhanced LS; reference eq_idx=5; Woodbury-equivalent solve",
        },
        "snrs": snrs,
        "results": curves,
        "errors": counts,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nbits", type=int, required=True)
    ap.add_argument("--cp", choices=("long", "short"), required=True)
    ap.add_argument("--channel", default="ETU", choices=("flat", "EPA", "EVA", "ETU"))
    ap.add_argument("--snrs", default="0,4,8,12,16,20,24,28")
    ap.add_argument("--seed-id", type=int, required=True)
    ap.add_argument("--slots", type=int, default=2000)
    ap.add_argument("--cp-batch", type=int, default=100)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    cfg = OfdmConfig(nbits=args.nbits, long_cp=(args.cp == "long"))
    snrs = [float(v) for v in args.snrs.split(",")]
    result = evaluate(cfg, args.channel, snrs, args.seed_id,
                      args.slots, args.cp_batch)
    result["system"] = {
        "nbits": args.nbits,
        "cp": args.cp,
        "channel": args.channel,
        "N": cfg.K,
        "symbols_per_slot": cfg.N_SYM,
        "data_res_per_slot": cfg.D_slot,
    }
    os.makedirs(RESULTS, exist_ok=True)
    suffix = f"_{args.tag}" if args.tag else ""
    out = os.path.join(
        RESULTS,
        f"classical_b{args.nbits}_{args.cp}cp_{args.channel.lower()}"
        f"_seed{args.seed_id}{suffix}.json",
    )
    with open(out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print("saved", out)


if __name__ == "__main__":
    main()
