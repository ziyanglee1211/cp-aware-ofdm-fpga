"""Two-stage training of DCCN (paper Sec. V-B):

Stage 1: basic receiver on AWGN, fixed Es/N0 = 5*m dB.
Stage 2: frozen basic receiver, train equalizer on Rayleigh fading with
         mixed SNR: uniform {0,3,...,15} U {18,21,...,30} dB with P(SNR>=18) = 90%.

Usage:
  python train.py --stage 1
  python train.py --stage 2
"""

import argparse
import copy
import os
import time

import torch

from config import OfdmConfig
from ofdm import OfdmTxTorch, ChannelTorch
from receiver import BasicReceiver, DCCNReceiver, ce_loss, ber

DEV = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CKPT_DIR = os.path.join(os.path.dirname(__file__), 'ckpt')


def sample_mixed_snr(batch, dev):
    """Training SNR distribution of the reference implementation
    (ofdmreceiver_np_mp.py): support {0,3,...,27} dB with
    p=[.01,.01,.02,.02,.02,.02,.10,.50,.20,.10] (mid-SNR dominant)."""
    p = torch.tensor([.01, .01, .02, .02, .02, .02, .10, .50, .20, .10], device=dev)
    idx = torch.multinomial(p, batch, replacement=True)
    return idx.float() * 3.0


def apply_fading_awgn(ch, y, snr_db):
    """Apply per-slot static fading + per-sample AWGN. y [n,F,S], snr_db [n]."""
    n, F, S = y.shape
    x = y.reshape(n, -1)
    h = ch.sample_h(n)
    if ch.L == 1:
        yr = x * h
    else:
        Lx = x.shape[1]
        nfft = 1 << (Lx + ch.L - 2).bit_length()
        yr = torch.fft.ifft(torch.fft.fft(x, n=nfft, dim=1)
                            * torch.fft.fft(h, n=nfft, dim=1), dim=1)[:, :Lx]
    sigma2 = (10 ** (-snr_db / 10.0)).view(-1, 1)
    noise = (torch.randn_like(yr) + 1j * torch.randn_like(yr)) * (sigma2 / 2) ** 0.5
    return (yr + noise).reshape(n, F, S), h


def train_stage1(cfg, iters=1200, batch=512, snr_db=None, seed=0):
    torch.manual_seed(seed)
    tx = OfdmTxTorch(cfg, DEV)
    snr = snr_db if snr_db is not None else 5.0 * cfg.m

    model = BasicReceiver(cfg).to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=500, gamma=0.98)

    best, best_state, patience = 1.0, None, 0
    for it in range(1, iters + 1):
        y, labels, _ = tx.transmit(batch)
        sigma2 = 10 ** (-snr / 10.0)
        noise = (torch.randn_like(y) + 1j * torch.randn_like(y)) * ((sigma2 / 2) ** 0.5)
        soft = model(y + noise)
        loss = ce_loss(soft, labels)
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        with torch.no_grad():
            b = ber(soft, labels)
        if b < best - 1e-5:
            best, best_state, patience = b, copy.deepcopy(model.state_dict()), 0
        else:
            patience += 1
        if it % 50 == 0:
            print(f"[stage1] iter {it:5d} loss {loss.item():.4f} train-BER {b:.5f} best {best:.5f}", flush=True)
        if patience >= 200 and it >= 400:
            print(f"[stage1] early stop at {it}, best BER {best:.5f}")
            break
    model.load_state_dict(best_state)
    return model, best


def train_stage2(cfg, basic: BasicReceiver, models=("flat", "EPA", "EVA", "ETU"),
                 iters=4000, batch=512, seed=1, patience=200, warmup=0):
    """warmup: number of initial iterations trained on flat only (curriculum;
    helps m>1 escape the flat-only basin when the frozen classifier is sharp)."""
    torch.manual_seed(seed)
    tx = OfdmTxTorch(cfg, DEV)
    chans = {m: ChannelTorch(cfg, DEV, model=m) for m in models}

    basic = copy.deepcopy(basic).to(DEV)
    for p in basic.parameters():
        p.requires_grad_(False)
    model = DCCNReceiver(cfg, basic).to(DEV)

    opt = torch.optim.Adam(model.eq.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=500, gamma=0.98)

    best, best_state, patience_left = 1.0, None, 0
    t0 = time.time()
    for it in range(1, iters + 1):
        y, labels, _ = tx.transmit(batch)
        snr = sample_mixed_snr(batch, DEV)
        # alternate channel model across slots (paper V-B: flat/EPA/EVA/ETU
        # alternate in consecutive slots)
        active = ("flat",) if it <= warmup else models
        yr = torch.empty_like(y)
        for gi, ms in enumerate(active):
            idx = torch.arange(gi, batch, len(active), device=DEV)
            y_g, _ = apply_fading_awgn(chans[ms], y[idx], snr[idx])
            yr[idx] = y_g
        soft = model(yr)
        loss = ce_loss(soft, labels)
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        with torch.no_grad():
            b = ber(soft, labels)
        if it > warmup and b < best - 1e-5:
            best, best_state, patience_left = b, copy.deepcopy(model.state_dict()), 0
        elif it > warmup:
            patience_left += 1
        if it % 50 == 0:
            print(f"[stage2] iter {it:5d} loss {loss.item():.4f} train-BER {b:.5f} "
                  f"best {best:.5f} ({time.time()-t0:.0f}s)", flush=True)
        if patience_left >= patience and it >= 1000:
            print(f"[stage2] early stop at {it}, best BER {best:.5f}")
            break
    model.load_state_dict(best_state)
    return model, best


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", type=int, default=1)
    ap.add_argument("--nbits", type=int, default=1)
    ap.add_argument("--cp", default="long", choices=["long", "short"])
    ap.add_argument("--iters", type=int, default=0)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--patience", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=0)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument(
        "--suffix", default="",
        help="checkpoint suffix used for both stages, e.g. _pdpfix_seed1",
    )
    ap.add_argument("--channel", default="all", help="comma list: flat,EPA,EVA,ETU or 'all'")
    args = ap.parse_args()

    os.makedirs(CKPT_DIR, exist_ok=True)
    cfg = OfdmConfig(nbits=args.nbits, long_cp=(args.cp == "long"))
    print(cfg.summary(), "| device:", DEV)

    tag = f"b{args.nbits}_{args.cp}cp"
    suffix = args.suffix
    if args.stage == 1:
        iters = args.iters or 1200 * args.nbits
        model, best = train_stage1(
            cfg, iters=iters, batch=args.batch, seed=args.seed
        )
        out = f"{CKPT_DIR}/basic_{tag}{suffix}.pt"
        torch.save(model.state_dict(), out)
        print(f"stage1 done, best train BER {best:.5f} -> {out}")
    else:
        basic = BasicReceiver(cfg).to(DEV)
        basic_path = f"{CKPT_DIR}/basic_{tag}{suffix}.pt"
        basic.load_state_dict(torch.load(basic_path, map_location=DEV, weights_only=True))
        models = ["flat", "EPA", "EVA", "ETU"] if args.channel == "all" else args.channel.split(",")
        iters = args.iters or 4000 * args.nbits
        model, best = train_stage2(cfg, basic, models=models, iters=iters,
                                   batch=args.batch, seed=args.seed,
                                   patience=args.patience, warmup=args.warmup)
        out = f"{CKPT_DIR}/dccn_{tag}{suffix}.pt"
        torch.save(model.state_dict(), out)
        print(f"stage2 done, best train BER {best:.5f} -> {out}")
