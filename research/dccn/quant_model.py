"""Quantized DCCN receiver (QAT model, paper plan C2).

Mirrors receiver.py module-for-module so the float checkpoints load directly
(state_dict keys are identical; `delta` is new and loaded with strict=False).

All complex layers quantize weights per-output-channel (re/im separate) and
activations per-tensor (static calibrated scale). The equalizer's complex
division and RMS normalization use the fixed-point operators from quant.py.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from layers import ComplexLinearSN, ComplexLinear
from quant import (QConfig, ActQ, make_act_q, quant_real_weight,
                   quant_complex_weight, fake_quant, rms_norm_fixed,
                   cdiv_fixed, PWLATanh, qmax_of)


class ComplexLinearSNQ(nn.Module):
    """Quantized version of layers.ComplexLinearSN (shared-across-symbols
    complex transform, the DFT-like / IDFT-like layers)."""

    def __init__(self, s_len, n_len, qc: QConfig, bias=True):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(s_len, n_len, dtype=torch.complex64)
                                   * (s_len ** -0.5 / 2 ** 0.5))
        self.bias = (nn.Parameter(torch.zeros(n_len, dtype=torch.complex64))
                     if bias else None)
        self.qc = qc
        self.q_in = make_act_q(qc)
        self.q_out = make_act_q(qc)

    def forward(self, x):
        qc = self.qc
        if qc.quant_a:
            x = self.q_in(x)
        W = quant_complex_weight(self.weight, qc.wbits, out_axis=1) \
            if qc.quant_w else self.weight
        y = torch.einsum('bfs,sn->bfn', x, W)
        if self.bias is not None:
            y = y + self.bias
        return self.q_out(y) if qc.quant_a else y


class ComplexLinearQ(nn.Module):
    """Quantized version of layers.ComplexLinear (dense complex layer)."""

    def __init__(self, in_features, out_features, qc: QConfig, bias=True):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features,
                                               dtype=torch.complex64)
                                   * (in_features ** -0.5 / 2 ** 0.5))
        self.bias = (nn.Parameter(torch.zeros(out_features, dtype=torch.complex64))
                     if bias else None)
        self.qc = qc
        self.q_in = make_act_q(qc)
        self.q_out = make_act_q(qc)

    def forward(self, x):
        qc = self.qc
        if qc.quant_a:
            x = self.q_in(x)
        W = quant_complex_weight(self.weight, qc.wbits, out_axis=0) \
            if qc.quant_w else self.weight
        y = x @ W.T
        if self.bias is not None:
            y = y + self.bias
        return self.q_out(y) if qc.quant_a else y


class LinearQ(nn.Module):
    """Quantized real-valued linear layer (chest subnet, classifier)."""

    def __init__(self, in_features, out_features, qc: QConfig, bias=True):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features)
                                   * in_features ** -0.5)
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None
        self.qc = qc
        self.q_in = ActQ(qc.abits)
        self.q_out = ActQ(qc.abits)

    def forward(self, x):
        qc = self.qc
        if qc.quant_a:
            x = self.q_in(x)
        W = quant_real_weight(self.weight, qc.wbits) if qc.quant_w else self.weight
        y = F.linear(x, W, self.bias)
        return self.q_out(y) if qc.quant_a else y


class QBasicReceiver(nn.Module):
    """Quantized BasicReceiver (receiver.BasicReceiver)."""

    def __init__(self, cfg, qc: QConfig):
        super().__init__()
        self.cfg, self.qc = cfg, qc
        self.dft_like = ComplexLinearSNQ(cfg.S, cfg.K, qc)
        self.extract = ComplexLinearQ(cfg.N_SYM * cfg.K, cfg.D_slot, qc)
        self.classifier = LinearQ(4, 2 * cfg.m, qc)
        self.q_norm = make_act_q(qc)

    def forward(self, y):
        cfg = self.cfg
        y = rms_norm_fixed(y, self.qc)
        y = self.q_norm(y)
        Y = self.dft_like(y)
        d = self.extract(Y.reshape(y.shape[0], -1))
        iq = torch.stack([d.real, d.imag], dim=-1)
        feat = torch.cat([iq, F.leaky_relu(iq)], dim=-1)
        logits = F.leaky_relu(self.classifier(feat))
        logits = logits.view(-1, cfg.D_slot, cfg.m, 2)
        return torch.softmax(logits, dim=-1)


class QEqualizer(nn.Module):
    """Quantized Equalizer (receiver.Equalizer).

    Same structure; the tanh subnet uses PWLA, the complex division and RMS
    normalization use fixed-point operators, and delta is learnable.
    """

    BOTTLENECK = 112

    def __init__(self, cfg, qc: QConfig):
        super().__init__()
        self.cfg, self.qc = cfg, qc
        F_, N, S = cfg.N_SYM, cfg.K, cfg.S
        self.e1 = ComplexLinearSNQ(S, N, qc)
        self.e2 = ComplexLinearSNQ(N, N, qc)
        b = self.BOTTLENECK
        self.c_in = LinearQ(F_ * N * 2, b, qc)
        self.c_res = nn.ModuleList([LinearQ(b, b, qc) for _ in range(6)])
        self.c_out = LinearQ(6 * b, F_ * N * 2, qc)
        self.i1 = ComplexLinearSNQ(N, N, qc)
        self.i2 = ComplexLinearSNQ(N, S, qc)
        self.delta = nn.Parameter(torch.tensor(0.1))
        self.act = PWLATanh(qc.pwla_k, qc.pwla_R) if qc.pwla_k > 0 else nn.Identity()
        self.q_norm = make_act_q(qc)
        self.q_div = make_act_q(qc)

    def forward(self, y):
        cfg = self.cfg
        qc = self.qc
        y = rms_norm_fixed(y, qc)
        y = self.q_norm(y)
        B = y.shape[0]
        Y = self.e2(self.e1(y))
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
        X = self.q_div(X) if qc.quant_a else X
        return self.i2(self.i1(X)), Y, H


class QDCCN(nn.Module):
    """Full quantized receiver = QEqualizer + QBasicReceiver."""

    def __init__(self, cfg, qc: QConfig):
        super().__init__()
        self.cfg = cfg
        self.qc = qc
        self.eq = QEqualizer(cfg, qc)
        self.basic = QBasicReceiver(cfg, qc)

    def load_float(self, basic_sd, dccn_sd):
        self.basic.load_state_dict(basic_sd, strict=False)
        eq_sd = {k[len("eq."):]: v for k, v in dccn_sd.items() if k.startswith("eq.")}
        self.eq.load_state_dict(eq_sd, strict=False)

    def forward(self, y):
        x_hat, _, _ = self.eq(y)
        return self.basic(x_hat)

    @torch.no_grad()
    def calibrate(self, tx, chans, snr_fn, n_slots=400, rounds=2, dev=None):
        """Static activation-scale calibration: run mixed traffic through the
        network in train mode so every ActQ absorbs the running max.
        Round 1 collects scales on the (passthrough) float path, round 2
        re-collects on the quantized path."""
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
