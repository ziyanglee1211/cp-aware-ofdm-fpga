"""Evaluate an unstructured SVD low-rank DCCN baseline.

This is an inference-only control: each trained dense complex linear operator
is replaced by its rank-r truncated SVD, while the channel-estimation MLP and
classifier are kept unchanged.  It intentionally has no CP-aware Fourier
backbone, so it isolates the value of the physical prior from generic low-rank
compression.  Results are saved as JSON for the paper ablation table.
"""
import argparse
import json
import os

import numpy as np
import torch
import torch.nn as nn

from config import OfdmConfig
from ofdm import OfdmTxTorch, ChannelTorch
from receiver import BasicReceiver, ce_loss, ber, rms_norm
from train import apply_fading_awgn

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
HERE = os.path.dirname(os.path.abspath(__file__))
CKPT = os.path.join(HERE, "ckpt")
RES = os.path.join(HERE, "results")


class LowRankSN(nn.Module):
    def __init__(self, weight, bias, rank):
        super().__init__()
        out, inn = weight.shape
        r = min(int(rank), out, inn)
        u, s, vh = torch.linalg.svd(weight.detach().cpu(), full_matrices=False)
        self.v = nn.Parameter(vh[:r, :].clone(), requires_grad=False)
        self.us = nn.Parameter((u[:, :r] * s[:r]).clone(), requires_grad=False)
        self.bias = nn.Parameter(bias.detach().cpu().clone(), requires_grad=False)

    def forward(self, x):
        # ComplexLinear convention: x @ W.T + b, with W = US Vh.
        return (x @ self.v.T.to(x.dtype)) @ self.us.T.to(x.dtype) + self.bias.to(x.dtype)


class LowRankSNShared(nn.Module):
    def __init__(self, weight, bias, rank):
        super().__init__()
        # ComplexLinearSN stores W as [input, output] and evaluates x @ W.
        inn, out = weight.shape
        r = min(int(rank), out, inn)
        u, s, vh = torch.linalg.svd(weight.detach().cpu(), full_matrices=False)
        # W = (U*S) @ Vh; x @ (U*S) @ Vh.
        self.us = nn.Parameter((u[:, :r] * s[:r]).clone(), requires_grad=False)
        self.vh = nn.Parameter(vh[:r, :].clone(), requires_grad=False)
        self.bias = nn.Parameter(bias.detach().cpu().clone(), requires_grad=False)

    def forward(self, x):
        return torch.einsum("bfi,ir->bfr", x, self.us.to(x.dtype)).matmul(
            self.vh.to(x.dtype)
        ) + self.bias.to(x.dtype)


class SVDLowRankDCCN(nn.Module):
    def __init__(self, cfg, basic_sd, dccn_sd, rank):
        super().__init__()
        self.cfg = cfg
        self.rank = rank
        # Basic path: preserve classifier exactly, approximate both complex maps.
        self.basic_dft = LowRankSNShared(
            basic_sd["dft_like.weight"], basic_sd["dft_like.bias"], rank
        )
        self.basic_extract = LowRankSN(
            basic_sd["extract.weight"], basic_sd["extract.bias"], rank
        )
        self.basic_classifier = nn.Linear(4, 2 * cfg.m)
        self.basic_classifier.load_state_dict(
            {k[len("classifier."):]: v for k, v in basic_sd.items() if k.startswith("classifier.")}
        )

        # Equalizer path: use the original real-valued subnet unchanged.
        self.e1 = LowRankSNShared(dccn_sd["eq.e1.weight"], dccn_sd["eq.e1.bias"], rank)
        self.e2 = LowRankSNShared(dccn_sd["eq.e2.weight"], dccn_sd["eq.e2.bias"], rank)
        self.i1 = LowRankSNShared(dccn_sd["eq.i1.weight"], dccn_sd["eq.i1.bias"], rank)
        self.i2 = LowRankSNShared(dccn_sd["eq.i2.weight"], dccn_sd["eq.i2.bias"], rank)
        self.c_in = nn.Linear(7 * 64 * 2, 112)
        self.c_in.load_state_dict({k[len("eq.c_in."):]: v for k, v in dccn_sd.items() if k.startswith("eq.c_in.")})
        self.c_res = nn.ModuleList([nn.Linear(112, 112) for _ in range(6)])
        for i, lin in enumerate(self.c_res):
            lin.load_state_dict({k[len(f"eq.c_res.{i}."):]: v for k, v in dccn_sd.items() if k.startswith(f"eq.c_res.{i}.")})
        self.c_out = nn.Linear(6 * 112, 7 * 64 * 2)
        self.c_out.load_state_dict({k[len("eq.c_out."):]: v for k, v in dccn_sd.items() if k.startswith("eq.c_out.")})
        self.to(DEV).eval()

    def forward(self, y):
        cfg = self.cfg
        y = rms_norm(y)
        Y = self.e2(self.e1(y))
        q = torch.cat([Y.real, Y.imag], dim=-1).reshape(y.shape[0], -1)
        h = self.c_in(q)
        hs, x = [], h
        for lin in self.c_res:
            hs.append(torch.tanh(lin(x)))
            x = (hs[-2] if len(hs) >= 2 else h) - hs[-1]
        Hq = self.c_out(torch.cat(hs, dim=-1)).view(y.shape[0], cfg.N_SYM, cfg.K, 2)
        H = torch.complex(Hq[..., 0], Hq[..., 1])
        X = Y * H.conj() / (H.abs().square() + 0.1)
        xhat = self.i2(self.i1(X))
        z = rms_norm(xhat)
        D = self.basic_extract(self.basic_dft(z).reshape(y.shape[0], -1))
        iq = torch.stack([D.real, D.imag], dim=-1)
        feat = torch.cat([iq, torch.nn.functional.leaky_relu(iq)], dim=-1)
        logits = torch.nn.functional.leaky_relu(self.basic_classifier(feat))
        logits = logits.view(-1, cfg.D_slot, cfg.m, 2)
        return torch.softmax(logits, dim=-1)


def evaluate(model, cfg, channels, snrs, slots, batch, seed):
    tx = OfdmTxTorch(cfg, DEV)
    out = {}
    for ch_name in channels:
        ch = ChannelTorch(cfg, DEV, model=ch_name)
        out[ch_name] = []
        for snr in snrs:
            torch.manual_seed(seed + int(snr * 10))
            errs = bits = 0
            done = 0
            while done < slots:
                n = min(batch, slots - done)
                y, labels, _ = tx.transmit(n)
                sv = torch.full((n,), float(snr), device=DEV)
                yr, _ = apply_fading_awgn(ch, y, sv)
                with torch.no_grad():
                    pred = (model(yr)[..., 1] > 0.5).float()
                errs += int((pred != labels).sum())
                bits += labels.numel()
                done += n
            value = errs / bits
            out[ch_name].append(value)
            print(f"rank={model.rank:3d} {ch_name:3s} {snr:4.0f} dB BER {value:.6e} ({slots} slots)", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nbits", type=int, default=1)
    ap.add_argument("--cp", choices=["long", "short"], default="long")
    ap.add_argument("--ranks", default="8,16,32")
    ap.add_argument("--channels", default="EVA,ETU")
    ap.add_argument("--snrs", default="12,20,24,28")
    ap.add_argument("--slots", type=int, default=500)
    ap.add_argument("--batch", type=int, default=100)
    ap.add_argument("--seed", type=int, default=20260913)
    ap.add_argument("--tagname", default="_svd_lowrank")
    args = ap.parse_args()
    cfg = OfdmConfig(nbits=args.nbits, long_cp=args.cp == "long")
    tag = f"b{args.nbits}_{args.cp}cp"
    basic_sd = torch.load(os.path.join(CKPT, f"basic_{tag}.pt"), map_location="cpu", weights_only=True)
    dccn_sd = torch.load(os.path.join(CKPT, f"dccn_{tag}.pt"), map_location="cpu", weights_only=True)
    channels = [x.strip() for x in args.channels.split(",") if x.strip()]
    snrs = [float(x) for x in args.snrs.split(",")]
    all_results = {}
    for rank in [int(x) for x in args.ranks.split(",")]:
        model = SVDLowRankDCCN(cfg, basic_sd, dccn_sd, rank)
        all_results[f"svd_r{rank}"] = evaluate(model, cfg, channels, snrs, args.slots, args.batch, args.seed)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    os.makedirs(RES, exist_ok=True)
    path = os.path.join(RES, f"svd_lowrank_{tag}{args.tagname}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"tag": tag, "ranks": args.ranks, "channels": channels, "snrs": snrs, "slots": args.slots, "results": all_results}, f, indent=2)
    print("saved", path)


if __name__ == "__main__":
    main()
