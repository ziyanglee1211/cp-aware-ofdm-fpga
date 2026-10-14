"""Train a generic (non-physical) complex low-rank DCCN control.

This uses the same mixed-channel/mixed-SNR protocol as struct_train.py.  The
only architectural change is replacing every complex linear operator with a
trainable rank-r factorization initialized by truncated SVD; no CP-aware
Fourier backbone is used.
"""
import argparse
import copy
import os
import time

import torch

from config import OfdmConfig
from eval_svd_lowrank import SVDLowRankDCCN, DEV, CKPT
from ofdm import OfdmTxTorch, ChannelTorch
from receiver import ce_loss, ber
from train import apply_fading_awgn, sample_mixed_snr


def run(args):
    cfg = OfdmConfig(nbits=args.nbits, long_cp=args.cp == "long")
    tag = f"b{args.nbits}_{args.cp}cp"
    src = args.source_suffix
    basic_sd = torch.load(os.path.join(CKPT, f"basic_{tag}{src}.pt"), map_location="cpu", weights_only=True)
    dccn_sd = torch.load(os.path.join(CKPT, f"dccn_{tag}{src}.pt"), map_location="cpu", weights_only=True)
    model = SVDLowRankDCCN(cfg, basic_sd, dccn_sd, args.rank).to(DEV)
    for p in model.parameters():
        p.requires_grad_(True)
    # Keep the model within the generic low-rank hypothesis: all factors are
    # trained, while the real channel subnet/classifier is also fine-tuned.
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    tx = OfdmTxTorch(cfg, DEV)
    chans = {m: ChannelTorch(cfg, DEV, model=m) for m in ("flat", "EPA", "EVA", "ETU")}
    best, best_state, stale = 1.0, None, 0
    t0 = time.time()
    model.train()
    for it in range(1, args.iters + 1):
        y, labels, _ = tx.transmit(args.batch)
        snr = sample_mixed_snr(args.batch, DEV)
        yr = torch.empty_like(y)
        for gi, name in enumerate(chans):
            idx = torch.arange(gi, args.batch, len(chans), device=DEV)
            yr[idx], _ = apply_fading_awgn(chans[name], y[idx], snr[idx])
        soft = model(yr)
        loss = ce_loss(soft, labels)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        with torch.no_grad():
            b = ber(soft, labels)
        if b < best - 1e-5:
            best, best_state, stale = b, copy.deepcopy(model.state_dict()), 0
        else:
            stale += 1
        if it == 1 or it % args.log_every == 0:
            print(f"[svd-r{args.rank}] iter {it:4d} loss {loss.item():.4f} BER {b:.5f} best {best:.5f} ({time.time()-t0:.1f}s)", flush=True)
        if stale >= args.patience:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    out = os.path.join(CKPT, f"svd_{tag}_r{args.rank}_seed{args.seed}.pt")
    torch.save(model.state_dict(), out)
    print(f"saved {out}; best_train_ber={best:.6f}")
    return model


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--rank", type=int, required=True)
    ap.add_argument("--nbits", type=int, default=1)
    ap.add_argument("--cp", choices=["long", "short"], default="long")
    ap.add_argument("--iters", type=int, default=800)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--patience", type=int, default=250)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--source-suffix", default="",
                    help="checkpoint suffix of the dense source model, "
                         "e.g. _pdpfix_seed1 (empty = pre-correction checkpoint)")
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    run(args)
