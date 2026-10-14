"""Quantized structured DCCN receiver (paper plan C1 x C2 joint).

Quantized counterpart of struct_model.StructuredDCCN.  Hardware contract:

  * CP-DFT / CP-IDFT backbones are FIXED and treated as EXACT: they map to a
    streaming FFT/IFFT IP (block-floating-point), so no fake-quant here.
  * Everything trainable around the backbone quantizes like quant_model.py:
    weights symmetric per-output-channel (re/im separate, INT wbits),
    activations per-tensor static scales (INT abits, calibrated).
  * Per-RE gains g, mixing diagonals and complex bias vectors use a single
    per-tensor complex scale each (they are LUT content in hardware, not
    matrix operands, so per-channel scaling buys nothing).
  * rms-norm / complex division / tanh use the fixed-point operators of
    quant.py (identical to the dense QAT model, so C2 results carry over).

Checkpoints: loads struct_{tag}_{name}.pt state dicts directly (the quant
model adds only quantizer buffers, loaded with strict=False).

Usage (PTQ smoke test):
  python struct_quant_model.py --tag b1_longcp --rank 0
"""

import argparse
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import OfdmConfig
from quant import (QConfig, make_act_q, quant_complex_weight, fake_quant,
                   rms_norm_fixed, cdiv_fixed, PWLATanh, qmax_of)
from quant_model import LinearQ
from struct_model import cp_dft


def quant_complex_vector(v, bits):
    """Per-tensor complex vector quantization (re/im separate scales)."""
    sr = v.real.abs().amax().clamp_min(1e-8) / qmax_of(bits)
    si = v.imag.abs().amax().clamp_min(1e-8) / qmax_of(bits)
    return torch.complex(fake_quant(v.real, sr, bits),
                         fake_quant(v.imag, si, bits))


def cp_fold_fft(y, cp):
    """Unitary 64-point FFT after folding the cyclic prefix into the tail."""
    if cp:
        body = torch.cat([y[..., cp:-cp], y[..., -cp:] + y[..., :cp]], dim=-1)
    else:
        body = y
    return torch.fft.fft(body, dim=-1, norm='ortho')


def cp_expand_ifft(y, cp):
    """Unitary IFFT followed by cyclic-prefix replication."""
    body = torch.fft.ifft(y, dim=-1, norm='ortho')
    return torch.cat([body[..., -cp:], body], dim=-1) if cp else body


class StructuredMixQ(nn.Module):
    """Quantized StructuredMix: dense or (diag + low-rank) N x N complex.

    Dense:      y = x @ Q(W)                     W per-output-channel (col)
    Diag+LR:    y = x * Q(d) + (x @ Q(U)) @ Q(V) intermediate re-quantized
    """

    def __init__(self, n, rank=0, dense=True, qc: QConfig = None):
        super().__init__()
        self.n, self.rank, self.dense = n, rank, dense
        self.qc = qc
        if dense:
            self.weight = nn.Parameter(torch.zeros(n, n, dtype=torch.complex64))
        else:
            self.diag = nn.Parameter(torch.zeros(n, dtype=torch.complex64))
            self.u = nn.Parameter(torch.zeros(n, rank, dtype=torch.complex64))
            self.v = nn.Parameter(torch.zeros(rank, n, dtype=torch.complex64))
            self.q_mid = make_act_q(qc)
        self.q_in = make_act_q(qc)
        self.q_out = make_act_q(qc)

    def forward(self, x):
        qc = self.qc
        if qc.quant_a:
            x = self.q_in(x)
        if self.dense:
            W = quant_complex_weight(self.weight, qc.wbits, out_axis=1) \
                if qc.quant_w else self.weight
            y = x @ W.to(x.dtype)
        else:
            d = quant_complex_vector(self.diag, qc.wbits) if qc.quant_w else self.diag
            U = quant_complex_weight(self.u, qc.wbits, out_axis=1) \
                if qc.quant_w else self.u
            V = quant_complex_weight(self.v, qc.wbits, out_axis=1) \
                if qc.quant_w else self.v
            t = x @ U.to(x.dtype)
            if qc.quant_a:
                t = self.q_mid(t)
            y = x * d.to(x.dtype) + t @ V.to(x.dtype)
        return self.q_out(y) if qc.quant_a else y


class StructuredBasicQ(nn.Module):
    """Quantized StructuredBasic: exact CP-DFT + INT8 per-RE gains."""

    def __init__(self, cfg, rank, qc: QConfig):
        super().__init__()
        self.cfg, self.qc = cfg, qc
        self.transform_backend = 'matrix'
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
            self.q_mid = make_act_q(qc)
        self.classifier = LinearQ(4, 2 * cfg.m, qc)
        self.q_norm = make_act_q(qc)
        self.q_fft = make_act_q(qc)
        self.q_out = make_act_q(qc)

    def forward(self, y):
        cfg, qc = self.cfg, self.qc
        B = y.shape[0]
        y = rms_norm_fixed(y, qc)
        if qc.quant_a:
            y = self.q_norm(y)
        Y = (cp_fold_fft(y, cfg.CP) if self.transform_backend == 'fft'
             else torch.einsum('bfs,sn->bfn', y, self.Fd.to(y.dtype)))
        if qc.quant_a:
            Y = self.q_fft(Y)
        g = quant_complex_vector(self.g, qc.wbits) if qc.quant_w else self.g
        d = Y[:, self.f_idx, self.k_idx] * g.to(y.dtype) + self.bias.to(y.dtype)
        if self.rank > 0:
            A = quant_complex_weight(self.a, qc.wbits, out_axis=0) \
                if qc.quant_w else self.a
            Bm = quant_complex_weight(self.b, qc.wbits, out_axis=0) \
                if qc.quant_w else self.b
            t = y.reshape(B, -1) @ A.to(y.dtype).T
            if qc.quant_a:
                t = self.q_mid(t)
            d = d + t @ Bm.to(y.dtype).T
        if qc.quant_a:
            d = self.q_out(d)
        iq = torch.stack([d.real, d.imag], dim=-1)
        feat = torch.cat([iq, F.leaky_relu(iq)], dim=-1)
        logits = F.leaky_relu(self.classifier(feat))
        logits = logits.view(-1, cfg.D_slot, cfg.m, 2)
        return torch.softmax(logits, dim=-1)


class StructuredEqualizerQ(nn.Module):
    """Quantized StructuredEqualizer: exact FFT/IFFT ends + INT8 mixings."""

    def __init__(self, cfg, eq_rank, eq_dense, slim, qc: QConfig):
        super().__init__()
        self.cfg, self.qc = cfg, qc
        self.transform_backend = 'matrix'
        F_, N, S = cfg.N_SYM, cfg.K, cfg.S
        self.register_buffer('Fd', cp_dft(S, N, cfg.CP))
        self.register_buffer('FdcT', cp_dft(S, N, cfg.CP, conj=True).T)
        self.m_front = StructuredMixQ(N, eq_rank, eq_dense, qc)
        self.m_back = StructuredMixQ(N, eq_rank, eq_dense, qc)
        self.b_front = nn.Parameter(torch.zeros(N, dtype=torch.complex64))
        self.b_back = nn.Parameter(torch.zeros(S, dtype=torch.complex64))

        b = 112
        n_res = 2 if slim else 6
        self.c_in = LinearQ(F_ * N * 2, b, qc)
        self.c_res = nn.ModuleList([LinearQ(b, b, qc) for _ in range(n_res)])
        self.c_out = LinearQ(n_res * b, F_ * N * 2, qc)
        self.slim = slim
        self.delta = nn.Parameter(torch.tensor(0.1))
        self.act = PWLATanh(qc.pwla_k, qc.pwla_R) if qc.pwla_k > 0 else nn.Identity()
        self.q_norm = make_act_q(qc)
        self.q_fft = make_act_q(qc)
        self.q_div = make_act_q(qc)
        self.q_ifft = make_act_q(qc)

    def forward(self, y):
        cfg, qc = self.cfg, self.qc
        B = y.shape[0]
        y = rms_norm_fixed(y, qc)
        if qc.quant_a:
            y = self.q_norm(y)
        Yf = (cp_fold_fft(y, cfg.CP) if self.transform_backend == 'fft'
              else torch.einsum('bfs,sn->bfn', y, self.Fd.to(y.dtype)))
        if qc.quant_a:
            Yf = self.q_fft(Yf)
        Y = self.m_front(Yf) + self.b_front.to(y.dtype)
        q = torch.cat([Y.real, Y.imag], dim=-1).reshape(B, -1)
        h = self.c_in(q)
        hs = []
        x = h
        for lin in self.c_res:
            hs.append(self.act(lin(x)) if qc.pwla_k > 0 else torch.tanh(lin(x)))
            x = (hs[-2] if len(hs) >= 2 else h) - hs[-1]
        z = torch.cat(hs, dim=-1)
        Hq = self.c_out(z).view(B, cfg.N_SYM, cfg.K, 2)
        H = torch.complex(Hq[..., 0], Hq[..., 1])
        X = cdiv_fixed(Y, H, self.delta.abs(), qc)
        if qc.quant_a:
            X = self.q_div(X)
        t = self.m_back(X)
        if self.transform_backend == 'fft':
            x_hat = cp_expand_ifft(t, cfg.CP) + self.b_back.to(t.dtype)
        else:
            x_hat = t @ self.FdcT.to(t.dtype) + self.b_back.to(t.dtype)
        if qc.quant_a:
            x_hat = self.q_ifft(x_hat)
        return x_hat, Y, H


class StructuredQDCCN(nn.Module):
    """Quantized structured receiver = StructuredEqualizerQ + StructuredBasicQ."""

    def __init__(self, cfg, rank=0, eq_rank=8, eq_dense=True, slim=False,
                 qc: QConfig = None):
        super().__init__()
        if qc is None:
            qc = QConfig()
        self.cfg, self.qc = cfg, qc
        self.eq = StructuredEqualizerQ(cfg, eq_rank, eq_dense, slim, qc)
        self.basic = StructuredBasicQ(cfg, rank, qc)

    def set_transform_backend(self, backend):
        """Select the original CP matrix or the FPGA-shaped FFT path.

        The default remains ``matrix`` so archived checkpoint results are
        unchanged.  ``fft`` is an explicit numerical-contract experiment:
        it changes float summation order near quantization boundaries.
        """
        if backend not in ('matrix', 'fft'):
            raise ValueError('transform backend must be matrix or fft')
        self.eq.transform_backend = backend
        self.basic.transform_backend = backend

    def load_struct(self, struct_sd):
        """Load a float structured checkpoint; quantizer buffers stay default."""
        self.load_state_dict(struct_sd, strict=False)

    def forward(self, y):
        x_hat, _, _ = self.eq(y)
        return self.basic(x_hat)

    @torch.no_grad()
    def calibrate(self, tx, chans, snr_fn, n_slots=400, rounds=2):
        for r in range(rounds):
            n_done = 0
            while n_done < n_slots:
                n = min(128, n_slots - n_done)
                y, _, _ = tx.transmit(n)
                snr = snr_fn(n)
                yr = torch.empty_like(y)
                for gi, (m, ch) in enumerate(chans.items()):
                    idx = torch.arange(gi, n, len(chans), device=y.device)
                    from train import apply_fading_awgn
                    yr[idx], _ = apply_fading_awgn(ch, y[idx], snr[idx])
                self(yr)
                n_done += n


def smoke(args):
    from ofdm import OfdmTxTorch, ChannelTorch
    from receiver import ber
    from struct_model import StructuredDCCN
    from train import apply_fading_awgn

    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cfg = OfdmConfig(nbits=int(args.tag[1]), long_cp=args.tag.endswith("longcp"))
    eq_dense = not args.diaglr
    name = f"s_r{args.rank}" + ("" if eq_dense else f"_d{args.eq_rank}") \
        + ("_slim" if args.slim else "")
    ckpt = os.path.join(os.path.dirname(__file__), 'ckpt',
                        f"struct_{args.tag}_{name}.pt")

    qc = QConfig(wbits=args.wbits, abits=args.abits)
    model = StructuredQDCCN(cfg, rank=args.rank, eq_rank=args.eq_rank,
                            eq_dense=eq_dense, slim=args.slim, qc=qc)
    model.load_struct(torch.load(ckpt, map_location='cpu', weights_only=True))
    model = model.to(dev)

    ref = StructuredDCCN(cfg, rank=args.rank, eq_rank=args.eq_rank,
                         eq_dense=eq_dense, slim=args.slim).to(dev)
    ref.load_state_dict(torch.load(ckpt, map_location=dev, weights_only=True))
    ref.eval()

    tx = OfdmTxTorch(cfg, dev)
    chans = {m: ChannelTorch(cfg, dev, model=m)
             for m in ("flat", "EPA", "EVA", "ETU")}
    from train import sample_mixed_snr
    model.train()
    model.calibrate(tx, chans, lambda n: sample_mixed_snr(n, dev), n_slots=256)
    model.eval()

    with torch.no_grad():
        for ms in ("EVA", "ETU"):
            for snr_db in (12.0, 24.0):
                torch.manual_seed(1234)
                y, labels, _ = tx.transmit(2000)
                snr = torch.full((2000,), snr_db, device=dev)
                yr, _ = apply_fading_awgn(chans[ms], y, snr)
                b_f = ber(ref(yr), labels)
                b_q = ber(model(yr), labels)
                print(f"  {ms:4s} {snr_db:4.0f} dB  struct-float BER {b_f:.5f}"
                      f"   struct-PTQ BER {b_q:.5f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="b1_longcp")
    ap.add_argument("--rank", type=int, default=0)
    ap.add_argument("--eq-rank", type=int, default=8)
    ap.add_argument("--diaglr", action="store_true")
    ap.add_argument("--slim", action="store_true")
    ap.add_argument("--wbits", type=int, default=8)
    ap.add_argument("--abits", type=int, default=8)
    smoke(ap.parse_args())
