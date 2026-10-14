"""Paired BER evaluation of structured (C1) receivers.

Same paired protocol as eval_qat.py (fixed seed per (channel, SNR) so every
configuration sees identical data/channel/noise) -- results are directly
comparable with the qat_*.json files produced earlier.

Usage:
  python eval_struct.py --nbits 1 --cp long --tagname _main \
    --configs float,s_r0,s_r8,s_r32,s_r128,s_r0_d8,s_r32_d8,s_r32_d8_slim
"""

import argparse
import json
import os

import numpy as np
import torch

from config import OfdmConfig
from ofdm import OfdmTxTorch, ChannelTorch
from receiver import BasicReceiver, DCCNReceiver
from struct_model import StructuredDCCN
from train import apply_fading_awgn

DEV = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
HERE = os.path.dirname(__file__)
CKPT_DIR = os.path.join(HERE, 'ckpt')
RES_DIR = os.path.join(HERE, 'results')


def build_model(s, cfg, tag):
    if s == "float":
        basic = BasicReceiver(cfg).to(DEV)
        basic.load_state_dict(torch.load(f"{CKPT_DIR}/basic_{tag}.pt",
                                         map_location=DEV, weights_only=True))
        model = DCCNReceiver(cfg, basic).to(DEV)
        model.load_state_dict(torch.load(f"{CKPT_DIR}/dccn_{tag}.pt",
                                         map_location=DEV, weights_only=True))
        model.eval()
        return model
    parts = s.split("_")
    assert parts[0] == "s" and parts[1].startswith("r"), s
    rank = int(parts[1][1:])
    eq_rank, eq_dense, slim = 8, True, False
    for p in parts[2:]:
        if p.startswith("d"):
            eq_rank, eq_dense = int(p[1:]), False
        elif p == "slim":
            slim = True
        else:
            raise ValueError(p)
    model = StructuredDCCN(cfg, rank=rank, eq_rank=eq_rank,
                           eq_dense=eq_dense, slim=slim).to(DEV)
    ckpt = f"{CKPT_DIR}/struct_{tag}_{s}.pt"
    model.load_state_dict(torch.load(ckpt, map_location=DEV, weights_only=True))
    model.eval()
    return model


def eval_ber(model, cfg, channel, snrs, min_errors=300, max_slots=20000,
             batch=1000, min_slots=2000, seed=20260904):
    tx = OfdmTxTorch(cfg, DEV)
    ch = ChannelTorch(cfg, DEV, model=channel)
    bers = []
    for snr in snrs:
        torch.manual_seed(seed + int(snr * 10))
        errs, bits, slots = 0, 0, 0
        while slots < max_slots and (errs < min_errors or slots < min_slots):
            n = min(batch, max_slots - slots)
            y, labels, _ = tx.transmit(n)
            snr_v = torch.full((n,), float(snr), device=DEV)
            yr, _ = apply_fading_awgn(ch, y, snr_v)
            with torch.no_grad():
                soft = model(yr)
                pred = (soft[..., 1] > 0.5).float()
                errs += int((pred != labels).sum())
            bits += labels.numel()
            slots += n
        bers.append(errs / bits)
        print(f"    {channel} {snr:4.0f} dB  BER {bers[-1]:.3e}  ({slots} slots)", flush=True)
    return bers


def snr_at_ber(snrs, bers, target):
    b = np.minimum.accumulate(np.asarray(bers, dtype=float))
    s = np.asarray(snrs, dtype=float)
    if b[-1] > target:
        return None
    for i in range(len(b) - 1):
        if b[i] > target >= b[i + 1]:
            lb, ln = np.log10(b[i]), np.log10(b[i + 1])
            w = (np.log10(target) - lb) / (ln - lb) if ln != lb else 0.0
            return float(s[i] + w * (s[i + 1] - s[i]))
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nbits", type=int, default=1)
    ap.add_argument("--cp", default="long", choices=["long", "short"])
    ap.add_argument("--configs", required=True)
    ap.add_argument("--channels", default="EVA,ETU")
    ap.add_argument("--snrs", default="0,4,8,12,16,20,24,28")
    ap.add_argument("--tber", type=float, default=2e-2)
    ap.add_argument("--tagname", default="")
    args = ap.parse_args()

    cfg = OfdmConfig(nbits=args.nbits, long_cp=(args.cp == "long"))
    tag = f"b{args.nbits}_{args.cp}cp"
    snrs = [float(s) for s in args.snrs.split(",")]
    channels = args.channels.split(",")
    configs = args.configs.split(",")

    os.makedirs(RES_DIR, exist_ok=True)
    results = {}
    for c in configs:
        print(f"== {c} ==")
        model = build_model(c, cfg, tag)
        results[c] = {}
        for chn in channels:
            results[c][chn] = eval_ber(model, cfg, chn, snrs)
        if "float" in results:
            s_ref = snr_at_ber(snrs, results["float"][channels[-1]], args.tber)
            s_c = snr_at_ber(snrs, results[c][channels[-1]], args.tber)
            if c != "float" and s_ref is not None and s_c is not None:
                print(f"  -> SNR@BER={args.tber:g}: {s_c:.1f} dB "
                      f"(float {s_ref:.1f} dB, loss {s_c - s_ref:+.2f} dB) [{channels[-1]}]")
        del model

    out = os.path.join(RES_DIR, f"struct_{tag}{args.tagname}.json")
    with open(out, "w") as f:
        json.dump({"snrs": snrs, "results": results}, f, indent=1)
    print("saved", out)


if __name__ == "__main__":
    main()
