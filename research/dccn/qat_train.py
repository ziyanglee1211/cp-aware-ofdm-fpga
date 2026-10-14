"""Quantization-aware fine-tuning of the DCCN receiver (paper plan C2 / E3).

Starts from the float checkpoint, calibrates static activation scales, then
fine-tunes ALL weights (equalizer + basic receiver) plus the equalizer delta
under the fixed-point emulation of quant.py.

Usage:
  python qat_train.py --nbits 1 --cp long --wbits 8 --abits 8
  python qat_train.py --nbits 1 --cp long --wbits 4 --abits 4 --iters 5000
Ablation overrides: --div shift|float --pwla 8|0 --cmode polar --no-w --no-a
"""

import argparse
import copy
import os
import time

import torch

from config import OfdmConfig
from ofdm import OfdmTxTorch, ChannelTorch
from receiver import BasicReceiver, DCCNReceiver, ce_loss, ber
from quant import QConfig
from quant_model import QDCCN
from train import apply_fading_awgn, sample_mixed_snr

DEV = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CKPT_DIR = os.path.join(os.path.dirname(__file__), 'ckpt')
CH_MODELS = ("flat", "EPA", "EVA", "ETU")


def build_qconfig(args):
    return QConfig(wbits=args.wbits, abits=args.abits,
                   quant_w=not args.no_w, quant_a=not args.no_a,
                   cmode=args.cmode, div=args.div, pwla_k=args.pwla,
                   rsqrt='float' if args.no_rsqrt else 'lut')


def train_qat(cfg, qc: QConfig, tag, iters=3000, batch=512, lr=1e-4,
              patience=600, seed=1):
    torch.manual_seed(seed)
    tx = OfdmTxTorch(cfg, DEV)
    chans = {m: ChannelTorch(cfg, DEV, model=m) for m in CH_MODELS}

    basic_sd = torch.load(f"{CKPT_DIR}/basic_{tag}.pt", map_location=DEV, weights_only=True)
    dccn_sd = torch.load(f"{CKPT_DIR}/dccn_{tag}.pt", map_location=DEV, weights_only=True)
    model = QDCCN(cfg, qc).to(DEV)
    model.load_float(basic_sd, dccn_sd)

    model.train()
    model.calibrate(tx, chans, lambda n: sample_mixed_snr(n, DEV), n_slots=256)
    print("calibrated, starting QAT")

    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=500, gamma=0.95)

    best, best_state, patience_left = 1.0, None, 0
    t0 = time.time()
    for it in range(1, iters + 1):
        y, labels, _ = tx.transmit(batch)
        snr = sample_mixed_snr(batch, DEV)
        yr = torch.empty_like(y)
        for gi, ms in enumerate(CH_MODELS):
            idx = torch.arange(gi, batch, len(CH_MODELS), device=DEV)
            yr[idx], _ = apply_fading_awgn(chans[ms], y[idx], snr[idx])
        soft = model(yr)
        loss = ce_loss(soft, labels)
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        with torch.no_grad():
            b = ber(soft, labels)
        if b < best - 1e-5:
            best, best_state, patience_left = b, copy.deepcopy(model.state_dict()), 0
        else:
            patience_left += 1
        if it % 100 == 0:
            print(f"[qat] iter {it:5d} loss {loss.item():.4f} train-BER {b:.5f} "
                  f"best {best:.5f} delta {model.eq.delta.abs().item():.4f} "
                  f"({time.time()-t0:.0f}s)", flush=True)
        if patience_left >= patience and it >= 500:
            print(f"[qat] early stop at {it}, best BER {best:.5f}")
            break
    model.load_state_dict(best_state)

    out = f"{CKPT_DIR}/qat_{tag}_{qc.name()}.pt"
    torch.save(model.state_dict(), out)
    print(f"[qat] saved {out} (best train BER {best:.5f})")

    # quick per-channel check at 12 / 24 dB
    model.eval()
    with torch.no_grad():
        for ms in CH_MODELS:
            for snr_db in (12.0, 24.0):
                y, labels, _ = tx.transmit(1000)
                snr = torch.full((1000,), snr_db, device=DEV)
                yr, _ = apply_fading_awgn(chans[ms], y, snr)
                soft = model(yr)
                print(f"  {ms:5s} {snr_db:4.0f} dB  BER {ber(soft, labels):.5f}")
    return model, best


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--nbits", type=int, default=1)
    ap.add_argument("--cp", default="long", choices=["long", "short"])
    ap.add_argument("--wbits", type=int, default=8)
    ap.add_argument("--abits", type=int, default=8)
    ap.add_argument("--div", default="lut_newton", choices=["lut_newton", "shift", "float"])
    ap.add_argument("--pwla", type=int, default=16, help="PWLA segments, 0 = float tanh")
    ap.add_argument("--cmode", default="cart", choices=["cart", "polar"])
    ap.add_argument("--no-w", action="store_true")
    ap.add_argument("--no-a", action="store_true")
    ap.add_argument("--no-rsqrt", action="store_true", help="float rsqrt (oracle)")
    ap.add_argument("--iters", type=int, default=3000)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--patience", type=int, default=600)
    args = ap.parse_args()

    cfg = OfdmConfig(nbits=args.nbits, long_cp=(args.cp == "long"))
    tag = f"b{args.nbits}_{args.cp}cp"
    qc = build_qconfig(args)
    print(cfg.summary(), "|", qc.name(), "| device:", DEV)
    train_qat(cfg, qc, tag, iters=args.iters, lr=args.lr, patience=args.patience)
