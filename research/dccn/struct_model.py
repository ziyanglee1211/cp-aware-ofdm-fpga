"""Structured reparameterization of the DCCN receiver (paper plan C1).

The corrected-PDP nine-checkpoint audit of the composed basic-receiver map
finds mean phase-invariant CP-DFT row matches of 0.963--0.987 per checkpoint,
with a global row minimum of 0.958 (see results/corrected_pdp_atom_match_audit.json).
This is an architectural diagnostic, not an isolated BER ablation.  The
equalizer front/back ends also motivate CP-DFT / CP-IDFT backbones.  We
therefore reparameterize:

  basic:   d = g * Sel(CP-DFT y) + b + B A y          (A [r, F*S], B [D, r])
  eq front: Y = (CP-DFT y) @ M + b_M                  (M dense or diag+lowrank)
  eq back:  x_hat = ((X @ Mp) @ CP-IDFT) + b_p        (Mp dense or diag+lowrank)

The CP-DFT / CP-IDFT backbones are FIXED (FFT hardware); only the small
corrections are trainable.  Init from the float checkpoint by least-squares /
SVD projection so training starts at the float solution's structured shadow.

Usage (init-quality smoke test):
  python struct_model.py --tag b1_longcp --rank 32
"""

import argparse
import math
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import OfdmConfig
from layers import rms_norm
from receiver import BasicReceiver

CKPT_DIR = os.path.join(os.path.dirname(__file__), 'ckpt')


def cp_dft(S, N, CP, conj=False):
    """CP-aggregating DFT matrix [S, N] (same convention as analyze_struct.py)."""
    s = torch.arange(S).unsqueeze(1)
    k = torch.arange(N).unsqueeze(0)
    n = (s - CP) % N
    sign = +1.0 if conj else -1.0
    return torch.exp(sign * 2j * np.pi * k * n / N) / (N ** 0.5)


class StructuredMix(nn.Module):
    """Dense or (diagonal + low-rank) complex N x N mixing.

    forward: x @ (diag(d) + U V) = x * d + (x @ U) @ V
    """

    def __init__(self, n, rank=0, dense=True):
        super().__init__()
        self.n, self.rank, self.dense = n, rank, dense
        if dense:
            self.weight = nn.Parameter(torch.zeros(n, n, dtype=torch.complex64))
        else:
            self.diag = nn.Parameter(torch.zeros(n, dtype=torch.complex64))
            self.u = nn.Parameter(torch.zeros(n, rank, dtype=torch.complex64))
            self.v = nn.Parameter(torch.zeros(rank, n, dtype=torch.complex64))

    def load_dense(self, W):
        """Init from a dense [n, n] matrix."""
        with torch.no_grad():
            if self.dense:
                self.weight.copy_(W)
            else:
                self.diag.copy_(torch.diagonal(W))
                R = W - torch.diag(torch.diagonal(W))
                U, S, Vh = torch.linalg.svd(R)
                r = self.rank
                self.u.copy_(U[:, :r] * S[:r])
                self.v.copy_(Vh[:r, :])

    def n_params(self):
        return self.n * self.n if self.dense else self.n + 2 * self.n * self.rank

    def forward(self, x):
        if self.dense:
            return x @ self.weight.to(x.dtype)
        return (x * self.diag.to(x.dtype)
                + (x @ self.u.to(x.dtype)) @ self.v.to(x.dtype))


class StructuredBasic(nn.Module):
    """Basic receiver with CP-DFT backbone + per-RE gain + low-rank residual."""

    def __init__(self, cfg, rank=0):
        super().__init__()
        self.cfg = cfg
        self.rank = rank
        F_, N, S, D = cfg.N_SYM, cfg.K, cfg.S, cfg.D_slot
        self.register_buffer('Fd', cp_dft(S, N, cfg.CP))
        pos = torch.tensor(cfg.data_pos)
        self.register_buffer('f_idx', pos[:, 0].long())
        self.register_buffer('k_idx', pos[:, 1].long())
        self.g = nn.Parameter(torch.ones(D, dtype=torch.complex64))
        self.bias = nn.Parameter(torch.zeros(D, dtype=torch.complex64))
        if rank > 0:
            self.a = nn.Parameter(torch.zeros(rank, F_ * S, dtype=torch.complex64))
            self.b = nn.Parameter(torch.zeros(D, rank, dtype=torch.complex64))
        self.classifier = nn.Linear(4, 2 * cfg.m)

    def init_from_basic(self, sd):
        """Least-squares gain + SVD residual init from the float basic state."""
        cfg = self.cfg
        F_, N, S, D = cfg.N_SYM, cfg.K, cfg.S, cfg.D_slot
        W_dft = sd['dft_like.weight'].cpu()          # [S, N]
        b_dft = sd['dft_like.bias'].cpu()            # [N]
        W_x = sd['extract.weight'].cpu()              # [D, F*N]
        b_x = sd['extract.bias'].cpu()                # [D]

        M = torch.einsum('sn,ifn->ifs', W_dft, W_x.view(D, F_, N)).reshape(D, F_ * S)
        Rsel = torch.zeros(D, F_ * S, dtype=torch.complex64)
        for i, (f_i, k_i) in enumerate(cfg.data_pos):
            Rsel[i, f_i * S:(f_i + 1) * S] = self.Fd[:, k_i]
        g = ((Rsel.conj() * M).sum(1) / Rsel.abs().square().sum(1))
        Res = M - g.unsqueeze(1) * Rsel
        bias_eff = torch.einsum('ifn,n->i', W_x.view(D, F_, N), b_dft) + b_x

        with torch.no_grad():
            self.g.copy_(g.to(self.g.dtype))
            self.bias.copy_(bias_eff.to(self.bias.dtype))
            if self.rank > 0:
                U, Sv, Vh = torch.linalg.svd(Res)
                r = self.rank
                self.b.copy_((U[:, :r] * Sv[:r]).to(self.b.dtype))
                self.a.copy_(Vh[:r, :].to(self.a.dtype))
            self.classifier.load_state_dict(
                {k[len('classifier.'):]: v for k, v in sd.items()
                 if k.startswith('classifier.')})
        rel = (Res.norm() / M.norm()).item()
        print(f"  [init-basic] rank {self.rank}: residual rel-Frobenius {rel:.3f}")

    def forward(self, y):
        cfg = self.cfg
        B = y.shape[0]
        y = rms_norm(y)
        Y = torch.einsum('bfs,sn->bfn', y, self.Fd.to(y.dtype))
        d = Y[:, self.f_idx, self.k_idx] * self.g.to(y.dtype) + self.bias.to(y.dtype)
        if self.rank > 0:
            d = d + (y.reshape(B, -1) @ self.a.to(y.dtype).T) @ self.b.to(y.dtype).T
        iq = torch.stack([d.real, d.imag], dim=-1)
        feat = torch.cat([iq, F.leaky_relu(iq)], dim=-1)
        logits = F.leaky_relu(self.classifier(feat))
        logits = logits.view(-1, cfg.D_slot, cfg.m, 2)
        return torch.softmax(logits, dim=-1)

    def n_complex(self):
        n = self.g.numel() + self.bias.numel()
        if self.rank > 0:
            n += self.a.numel() + self.b.numel()
        return n

    def n_macs(self):
        """Complex MACs/slot: FFT butterfly count + gains + residual."""
        cfg = self.cfg
        fft = cfg.N_SYM * (cfg.K // 2) * int(math.log2(cfg.K))
        res = self.rank * (cfg.N_SYM * cfg.S + cfg.D_slot) if self.rank > 0 else 0
        return fft + cfg.D_slot + res


class StructuredEqualizer(nn.Module):
    """Equalizer with fixed CP-DFT front / CP-IDFT back and small mixings."""

    def __init__(self, cfg, eq_rank=8, eq_dense=True, slim=False):
        super().__init__()
        self.cfg = cfg
        F_, N, S = cfg.N_SYM, cfg.K, cfg.S
        self.register_buffer('Fd', cp_dft(S, N, cfg.CP))
        self.register_buffer('FdcT', cp_dft(S, N, cfg.CP, conj=True).T)  # [N, S]
        self.m_front = StructuredMix(N, eq_rank, dense=eq_dense)
        self.m_back = StructuredMix(N, eq_rank, dense=eq_dense)
        self.b_front = nn.Parameter(torch.zeros(N, dtype=torch.complex64))
        self.b_back = nn.Parameter(torch.zeros(S, dtype=torch.complex64))

        b = 112
        n_res = 2 if slim else 6
        self.c_in = nn.Linear(F_ * N * 2, b)
        self.c_res = nn.ModuleList([nn.Linear(b, b) for _ in range(n_res)])
        self.c_out = nn.Linear(n_res * b, F_ * N * 2)
        self.slim = slim

    def init_from_eq(self, sd):
        """Projection init from the float equalizer state."""
        W_e = sd['eq.e1.weight'].cpu() @ sd['eq.e2.weight'].cpu()   # [S, N]
        W_i = sd['eq.i1.weight'].cpu() @ sd['eq.i2.weight'].cpu()   # [N, S]
        M0 = torch.linalg.pinv(self.Fd) @ W_e                        # [N, N]
        Mp0 = W_i @ torch.linalg.pinv(self.FdcT)                     # [N, N]
        b_f = sd['eq.e1.bias'].cpu() @ sd['eq.e2.weight'].cpu() + sd['eq.e2.bias'].cpu()
        b_b = sd['eq.i1.bias'].cpu() @ sd['eq.i2.weight'].cpu() + sd['eq.i2.bias'].cpu()
        with torch.no_grad():
            self.m_front.load_dense(M0)
            self.m_back.load_dense(Mp0)
            self.b_front.copy_(b_f.to(self.b_front.dtype))
            self.b_back.copy_(b_b.to(self.b_back.dtype))
        self.c_in.load_state_dict(
            {kk[len('eq.c_in.'):]: vv for kk, vv in sd.items()
             if kk.startswith('eq.c_in.')})
        for i, lin in enumerate(self.c_res):
            lin.load_state_dict(
                {kk[len(f'eq.c_res.{i}.'):]: vv for kk, vv in sd.items()
                 if kk.startswith(f'eq.c_res.{i}.')})
        wo = {kk[len('eq.c_out.'):]: vv.clone() for kk, vv in sd.items()
              if kk.startswith('eq.c_out.')}
        if self.slim:
            wo['weight'] = wo['weight'][:, :2 * 112].clone()   # keep first-2-hs slice
        self.c_out.load_state_dict(wo)
        relf = ((W_e - self.Fd @ M0).norm() / W_e.norm()).item()
        relb = ((W_i - Mp0 @ self.FdcT).norm() / W_i.norm()).item()
        print(f"  [init-eq] front residual {relf:.3f}  back residual {relb:.3f}")

    def forward(self, y):
        cfg = self.cfg
        y = rms_norm(y)
        B = y.shape[0]
        Yf = torch.einsum('bfs,sn->bfn', y, self.Fd.to(y.dtype))
        Y = self.m_front(Yf) + self.b_front.to(y.dtype)
        q = torch.cat([Y.real, Y.imag], dim=-1).reshape(B, -1)
        h = self.c_in(q)
        hs = []
        x = h
        for lin in self.c_res:
            hs.append(torch.tanh(lin(x)))
            x = (hs[-2] if len(hs) >= 2 else h) - hs[-1]
        z = torch.cat(hs, dim=-1)
        Hq = self.c_out(z).view(B, cfg.N_SYM, cfg.K, 2)
        H = torch.complex(Hq[..., 0], Hq[..., 1])
        X = Y * H.conj() / (H.abs().square() + 0.1)
        x_hat = self.m_back(X) @ self.FdcT.to(y.dtype) + self.b_back.to(y.dtype)
        return x_hat, Y, H

    def n_complex(self):
        return (self.m_front.n_params() + self.m_back.n_params()
                + self.b_front.numel() + self.b_back.numel())

    def n_macs(self):
        cfg = self.cfg
        F_, N = cfg.N_SYM, cfg.K
        fft = F_ * (N // 2) * int(math.log2(N))
        macs = 2 * fft
        for m in (self.m_front, self.m_back):
            if m.dense:
                macs += F_ * N * N
            else:
                macs += F_ * (N + 2 * N * m.rank)
        return macs


class StructuredDCCN(nn.Module):
    """eq(structured) + basic(structured), fully fine-tunable."""

    def __init__(self, cfg, rank=0, eq_rank=8, eq_dense=True, slim=False):
        super().__init__()
        self.cfg = cfg
        self.eq = StructuredEqualizer(cfg, eq_rank=eq_rank, eq_dense=eq_dense, slim=slim)
        self.basic = StructuredBasic(cfg, rank=rank)

    def init_from_float(self, basic_sd, dccn_sd):
        self.basic.init_from_basic(basic_sd)
        self.eq.init_from_eq(dccn_sd)

    def forward(self, y):
        x_hat, _, _ = self.eq(y)
        return self.basic(x_hat)

    def accounting(self):
        """Complex params / complex MACs per slot for the transform stack."""
        cfg = self.cfg
        pb, pe = self.basic.n_complex(), self.eq.n_complex()
        mb, me = self.basic.n_macs(), self.eq.n_macs()
        chest = (cfg.N_SYM * cfg.K * 2 * 112
                 + len(self.eq.c_res) * 112 * 112
                 + len(self.eq.c_res) * 112 * (cfg.N_SYM * cfg.K * 2))
        return {"basic_cx": pb, "eq_cx": pe, "cx_total": pb + pe,
                "basic_macs": mb, "eq_macs": me, "macs_total": mb + me,
                "chest_real": chest,
                "cx_compress": 166912 / (pb + pe),
                "mac_compress": 308224 / (mb + me)}


FLOAT_CX, FLOAT_MACS = 166912, 308224   # b1_longcp reference (E4 Tab.)


def smoke(args):
    tag = args.tag
    cfg = OfdmConfig(nbits=int(tag[1]), long_cp=tag.endswith("longcp"))
    basic_sd = torch.load(f"{CKPT_DIR}/basic_{tag}.pt", map_location='cpu', weights_only=True)
    dccn_sd = torch.load(f"{CKPT_DIR}/dccn_{tag}.pt", map_location='cpu', weights_only=True)

    model = StructuredDCCN(cfg, rank=args.rank, eq_rank=args.eq_rank,
                           eq_dense=not args.diaglr)
    model.init_from_float(basic_sd, dccn_sd)
    acc = model.accounting()
    print(f"structured rank={args.rank} eq_dense={not args.diaglr}:")
    for k, v in acc.items():
        print(f"  {k:14s} {v:10.1f}" if isinstance(v, float) else f"  {k:14s} {v:10d}")

    # float reference forward on identical input
    basic = BasicReceiver(cfg)
    basic.load_state_dict(basic_sd)
    from receiver import DCCNReceiver
    fl = DCCNReceiver(cfg, basic)
    fl.load_state_dict(dccn_sd, strict=False)

    torch.manual_seed(0)
    from ofdm import OfdmTxTorch, ChannelTorch
    from train import apply_fading_awgn
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tx = OfdmTxTorch(cfg, dev)
    ch = ChannelTorch(cfg, dev, model="EVA")
    y, labels, _ = tx.transmit(512)
    snr = torch.full((512,), 18.0, device=dev)
    yr, _ = apply_fading_awgn(ch, y, snr)
    fl, model = fl.to(dev), model.to(dev)
    with torch.no_grad():
        s_fl = fl(yr)
        s_st = model(yr)
    from receiver import ber
    print(f"float BER {ber(s_fl, labels):.5f}   structured(init) BER {ber(s_st, labels):.5f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="b1_longcp")
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--eq-rank", type=int, default=8)
    ap.add_argument("--diaglr", action="store_true")
    smoke(ap.parse_args())
