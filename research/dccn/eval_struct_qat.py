"""Paired BER evaluation of quantized structured receivers (C1 x C2).

Config strings:
  float                      dense float reference
  s_r0, s_r32_d8, ...        float structured reference (C1 checkpoints)
  ptq_<struct>_w{w}a{a}      post-training quantization of <struct> (calibrated
                             on the fly, fixed seed)
  qat_<struct>_w{w}a{a}      QAT checkpoint from struct_qat_train.py

Same paired protocol as eval_struct.py / eval_qat.py (identical seed per
(channel, SNR)), so results merge directly with the earlier JSONs.

Usage:
  python eval_struct_qat.py --nbits 1 --cp long --tagname _main \
    --configs float,s_r0,ptq_s_r0_w8a8,qat_s_r0_w8a8,qat_s_r32_d8_w8a8
"""

import argparse
import json
import os
import re

import numpy as np
import torch

from config import OfdmConfig
from ofdm import OfdmTxTorch, ChannelTorch
from receiver import BasicReceiver, DCCNReceiver
from struct_model import StructuredDCCN
from struct_quant_model import StructuredQDCCN
from train import apply_fading_awgn, sample_mixed_snr
from quant import QConfig

DEV = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
HERE = os.path.dirname(os.path.abspath(__file__))
CKPT_DIR = os.path.join(HERE, 'ckpt')
RES_DIR = os.path.join(HERE, 'results')
CH_MODELS = ("flat", "EPA", "EVA", "ETU")


def parse_struct(s):
    """'s_r32_d8_slim' -> (rank, eq_rank, eq_dense, slim)."""
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
    return rank, eq_rank, eq_dense, slim


def build_model(s, cfg, tag, tx, chans, checkpoint_suffix="",
                float_checkpoint_suffix=""):
    if s == "float":
        basic = BasicReceiver(cfg).to(DEV)
        basic.load_state_dict(torch.load(
                                         f"{CKPT_DIR}/basic_{tag}"
                                         f"{float_checkpoint_suffix}.pt",
                                         map_location=DEV, weights_only=True))
        model = DCCNReceiver(cfg, basic).to(DEV)
        model.load_state_dict(torch.load(
                                         f"{CKPT_DIR}/dccn_{tag}"
                                         f"{float_checkpoint_suffix}.pt",
                                         map_location=DEV, weights_only=True))
        model.eval()
        return model

    m = re.fullmatch(r"(ptq|qat)_(s_r\d+(?:_d\d+)?(?:_slim)?)_w(\d+)a(\d+)", s)
    if m:
        kind, sname, w, a = m.group(1), m.group(2), int(m.group(3)), int(m.group(4))
        rank, eq_rank, eq_dense, slim = parse_struct(sname)
        qc = QConfig(wbits=w, abits=a)
        model = StructuredQDCCN(cfg, rank=rank, eq_rank=eq_rank,
                                eq_dense=eq_dense, slim=slim, qc=qc).to(DEV)
        if kind == "qat":
            ckpt = (f"{CKPT_DIR}/sq_{tag}_{sname}_{qc.name()}"
                    f"{checkpoint_suffix}.pt")
            model.load_state_dict(torch.load(ckpt, map_location=DEV,
                                             weights_only=True))
        else:
            model.load_struct(torch.load(
                                         f"{CKPT_DIR}/struct_{tag}_{sname}"
                                         f"{checkpoint_suffix}.pt",
                                         map_location=DEV, weights_only=True))
            torch.manual_seed(777)
            model.train()
            model.calibrate(tx, chans, lambda n: sample_mixed_snr(n, DEV),
                            n_slots=256)
        model.eval()
        return model

    # float structured reference
    rank, eq_rank, eq_dense, slim = parse_struct(s)
    model = StructuredDCCN(cfg, rank=rank, eq_rank=eq_rank,
                           eq_dense=eq_dense, slim=slim).to(DEV)
    ckpt = f"{CKPT_DIR}/struct_{tag}_{s}{checkpoint_suffix}.pt"
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
        print(f"    {channel} {snr:4.0f} dB  BER {bers[-1]:.3e}  ({slots} slots)",
              flush=True)
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
    ap.add_argument("--checkpoint-suffix", default="")
    ap.add_argument(
        "--float-checkpoint-suffix", default="",
        help="suffix for the FP32 basic/DCCN checkpoints",
    )
    ap.add_argument("--eval-seed", type=int, default=20260904)
    ap.add_argument("--min-errors", type=int, default=300)
    ap.add_argument("--min-slots", type=int, default=2000)
    ap.add_argument("--max-slots", type=int, default=20000)
    ap.add_argument("--batch", type=int, default=1000)
    args = ap.parse_args()

    cfg = OfdmConfig(nbits=args.nbits, long_cp=(args.cp == "long"))
    tag = f"b{args.nbits}_{args.cp}cp"
    snrs = [float(s) for s in args.snrs.split(",")]
    channels = args.channels.split(",")
    configs = args.configs.split(",")

    tx = OfdmTxTorch(cfg, DEV)
    chans = {m: ChannelTorch(cfg, DEV, model=m) for m in CH_MODELS}

    os.makedirs(RES_DIR, exist_ok=True)
    results = {}
    for c in configs:
        print(f"== {c} ==")
        model = build_model(c, cfg, tag, tx, chans,
                            checkpoint_suffix=args.checkpoint_suffix,
                            float_checkpoint_suffix=args.float_checkpoint_suffix)
        results[c] = {}
        for chn in channels:
            results[c][chn] = eval_ber(
                model, cfg, chn, snrs,
                min_errors=args.min_errors,
                max_slots=args.max_slots,
                batch=args.batch,
                min_slots=args.min_slots,
                seed=args.eval_seed,
            )
        if "float" in results:
            s_ref = snr_at_ber(snrs, results["float"][channels[-1]], args.tber)
            s_c = snr_at_ber(snrs, results[c][channels[-1]], args.tber)
            if c != "float" and s_ref is not None and s_c is not None:
                print(f"  -> SNR@BER={args.tber:g}: {s_c:.1f} dB "
                      f"(float {s_ref:.1f} dB, loss {s_c - s_ref:+.2f} dB) "
                      f"[{channels[-1]}]")
        del model
        torch.cuda.empty_cache()

    out = os.path.join(RES_DIR, f"structq_{tag}{args.tagname}.json")
    with open(out, "w") as f:
        json.dump({
            "protocol": {
                "pdp_mode": chans["ETU"].pdp_mode,
                "pdp_note": "3GPP dB gains converted to normalized linear power",
                "checkpoint_suffix": args.checkpoint_suffix,
                "float_checkpoint_suffix": args.float_checkpoint_suffix,
                "eval_seed": args.eval_seed,
                "min_slots": args.min_slots,
                "max_slots": args.max_slots,
            },
            "snrs": snrs,
            "results": results,
        }, f, indent=1)
    print("saved", out)


if __name__ == "__main__":
    main()
