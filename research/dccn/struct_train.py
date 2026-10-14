"""Fine-tune the structured (reparameterized) DCCN receiver (paper plan C1).

Starts from the float checkpoint's structured projection (struct_model.py),
then fine-tunes all structured parameters end-to-end with CE loss under the
same mixed-channel/SNR traffic as qat_train.py (identical methodology).

Usage:
  python struct_train.py --nbits 1 --cp long --rank 0            # backbone only
  python struct_train.py --nbits 1 --cp long --rank 32 --diaglr  # eq diag+lowrank
  python struct_train.py --nbits 1 --cp long --rank 32 --diaglr --slim
"""

import argparse
import copy
import os
import time

import torch

from config import OfdmConfig
from ofdm import OfdmTxTorch, ChannelTorch
from receiver import ce_loss, ber
from struct_model import StructuredDCCN
from train import apply_fading_awgn, sample_mixed_snr

DEV = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CKPT_DIR = os.path.join(os.path.dirname(__file__), 'ckpt')
CH_MODELS = ("flat", "EPA", "EVA", "ETU")


def cfg_name(rank, eq_rank, eq_dense, slim):
    n = f"s_r{rank}"
    if not eq_dense:
        n += f"_d{eq_rank}"
    if slim:
        n += "_slim"
    return n


def train_struct(cfg, rank, eq_rank, eq_dense, slim, tag,
                 iters=3000, batch=512, lr=1e-4, patience=600, seed=1,
                 suffix="", source_suffix="", post_eval_slots=1000):
    torch.manual_seed(seed)
    name = cfg_name(rank, eq_rank, eq_dense, slim)
    tx = OfdmTxTorch(cfg, DEV)
    chans = {m: ChannelTorch(cfg, DEV, model=m) for m in CH_MODELS}

    basic_sd = torch.load(
        f"{CKPT_DIR}/basic_{tag}{source_suffix}.pt",
        map_location=DEV, weights_only=True,
    )
    dccn_sd = torch.load(
        f"{CKPT_DIR}/dccn_{tag}{source_suffix}.pt",
        map_location=DEV, weights_only=True,
    )
    model = StructuredDCCN(cfg, rank=rank, eq_rank=eq_rank, eq_dense=eq_dense, slim=slim)
    model.init_from_float(basic_sd, dccn_sd)
    model = model.to(DEV)
    acc = model.accounting()
    print(f"[{name}] cx params {acc['cx_total']} (x{acc['cx_compress']:.1f} vs float), "
          f"cx MACs {acc['macs_total']} (x{acc['mac_compress']:.1f})")

    model.train()
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
            print(f"[{name}] iter {it:5d} loss {loss.item():.4f} train-BER {b:.5f} "
                  f"best {best:.5f} ({time.time()-t0:.0f}s)", flush=True)
        if patience_left >= patience and it >= 500:
            print(f"[{name}] early stop at {it}, best BER {best:.5f}")
            break
    model.load_state_dict(best_state)

    out = f"{CKPT_DIR}/struct_{tag}_{name}{suffix}.pt"
    torch.save(model.state_dict(), out)
    print(f"[{name}] saved {out} (best train BER {best:.5f})")

    model.eval()
    with torch.no_grad():
        for ms in CH_MODELS:
            for snr_db in (12.0, 24.0):
                y, labels, _ = tx.transmit(post_eval_slots)
                snr = torch.full((post_eval_slots,), snr_db, device=DEV)
                yr, _ = apply_fading_awgn(chans[ms], y, snr)
                soft = model(yr)
                print(f"  {ms:5s} {snr_db:4.0f} dB  BER {ber(soft, labels):.5f}")
    return model, best


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--nbits", type=int, default=1)
    ap.add_argument("--cp", default="long", choices=["long", "short"])
    ap.add_argument("--rank", type=int, default=0, help="basic residual rank")
    ap.add_argument("--eq-rank", type=int, default=8, help="eq mixing low-rank")
    ap.add_argument("--diaglr", action="store_true", help="eq M = diag + lowrank")
    ap.add_argument("--slim", action="store_true", help="chest 6->2 residual layers")
    ap.add_argument("--iters", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--patience", type=int, default=600)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--post-eval-slots", type=int, default=1000)
    ap.add_argument(
        "--suffix", default="",
        help="checkpoint suffix, e.g. _seed2; defaults to empty for seed 1 "
             "and _seedN for other seeds",
    )
    ap.add_argument(
        "--source-suffix", default="",
        help="suffix of the FP32 basic/DCCN source checkpoints; kept separate "
             "so legacy multi-seed structured runs still load the shared FP32 model",
    )
    args = ap.parse_args()

    cfg = OfdmConfig(nbits=args.nbits, long_cp=(args.cp == "long"))
    tag = f"b{args.nbits}_{args.cp}cp"
    print(cfg.summary(), "| device:", DEV)
    suffix = args.suffix or ("" if args.seed == 1 else f"_seed{args.seed}")
    train_struct(cfg, args.rank, args.eq_rank, not args.diaglr, args.slim, tag,
                 iters=args.iters, batch=args.batch, lr=args.lr,
                 patience=args.patience, seed=args.seed, suffix=suffix,
                 source_suffix=args.source_suffix,
                 post_eval_slots=args.post_eval_slots)
