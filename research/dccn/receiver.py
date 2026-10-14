"""DCCN receivers in PyTorch (reproduction of Zhao et al., IEEE JSAC 2021, Fig. 5)."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from layers import ComplexLinear, ComplexLinearSN, rms_norm


class BasicReceiver(nn.Module):
    """Basic DCCN receiver: time-domain slot -> soft bits.

    1) (RMS norm) 2) DFT-like complex linear transform S->N (uses CP)
    3) complex dense C^{FN x D} extracting data REs
    4) classifier: [IQ | LeakyReLU(IQ)] -> dense R^{4 x 2m} -> LeakyReLU -> softmax
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.dft_like = ComplexLinearSN(cfg.S, cfg.K)
        self.extract = ComplexLinear(cfg.N_SYM * cfg.K, cfg.D_slot)
        self.classifier = nn.Linear(4, 2 * cfg.m)

    def forward(self, y):
        cfg = self.cfg
        y = rms_norm(y)
        Y = self.dft_like(y)                          # [B, F, N]
        d = self.extract(Y.reshape(y.shape[0], -1))   # [B, D] complex
        iq = torch.stack([d.real, d.imag], dim=-1)    # [B, D, 2]
        feat = torch.cat([iq, F.leaky_relu(iq)], dim=-1)  # [B, D, 4]
        logits = F.leaky_relu(self.classifier(feat))  # [B, D, 2m]
        logits = logits.view(-1, cfg.D_slot, cfg.m, 2)
        return torch.softmax(logits, dim=-1)          # soft bits [B, D, m, 2]


class Equalizer(nn.Module):
    """DCCN equalizer: time-domain slot -> equalized time-domain slot.

    DFT-like dense C^{S x N} + C-Conv N x N x 1 -> Y (complex, learnable)
    channel-estimation subnet: real-valued tanh network with a low-dimensional
    bottleneck (112 = F*P*2) and chained-differential residual dense layers,
    following the reference implementation (dev/py/model.py)
    complex division X = Y * conj(H) / (|H|^2 + delta)
    IDFT-like C-Conv + dense C^{N x S} (add CP back), all complex-valued.
    """

    BOTTLENECK = 112   # n_sym * P * m_iq = 7 * 8 * 2, as in reference code

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        F_, N, S = cfg.N_SYM, cfg.K, cfg.S
        self.e1 = ComplexLinearSN(S, N)               # dense C^{S x N}
        self.e2 = ComplexLinearSN(N, N)               # C-Conv N x N x 1
        b = self.BOTTLENECK
        self.c_in = nn.Linear(F_ * N * 2, b)
        self.c_res = nn.ModuleList([nn.Linear(b, b) for _ in range(6)])
        self.c_out = nn.Linear(6 * b, F_ * N * 2)
        self.i1 = ComplexLinearSN(N, N)               # IDFT-like C-Conv
        self.i2 = ComplexLinearSN(N, S)               # add back CP

    def forward(self, y):
        cfg = self.cfg
        y = rms_norm(y)
        B = y.shape[0]
        Y = self.e2(self.e1(y))                       # [B, F, N]
        # channel-estimation subnet (real-valued, tanh, differential residual)
        q = torch.cat([Y.real, Y.imag], dim=-1).reshape(B, -1)
        h = self.c_in(q)
        hs = []
        x = h
        for lin in self.c_res:
            hs.append(torch.tanh(lin(x)))
            x = (hs[-2] if len(hs) >= 2 else h) - hs[-1]
        z = torch.cat(hs, dim=-1)                     # [B, 6*112]
        Hq = self.c_out(z).view(B, cfg.N_SYM, cfg.K, 2)
        H = torch.complex(Hq[..., 0], Hq[..., 1])     # [B, F, N]
        X = Y * H.conj() / (H.abs().square() + 0.1)   # stabilized complex division
        x_hat = self.i2(self.i1(X))                   # [B, F, S] with CP
        return x_hat, Y, H


class DCCNReceiver(nn.Module):
    """Full receiver = equalizer + (optionally frozen) basic receiver."""

    def __init__(self, cfg, basic: BasicReceiver):
        super().__init__()
        self.cfg = cfg
        self.eq = Equalizer(cfg)
        self.basic = basic

    def forward(self, y):
        x_hat, _, _ = self.eq(y)
        return self.basic(x_hat)


def ce_loss(soft_bits, labels):
    """Soft bits [B,D,m,2] vs 0/1 labels [B,D,m] -> mean cross entropy."""
    p = soft_bits.clamp_min(1e-9)
    lp = torch.log(torch.stack([p[..., 0], p[..., 1]], dim=-1))
    ll = lp.gather(-1, labels.unsqueeze(-1).long()).squeeze(-1)
    return -ll.mean()


def ber(soft_bits, labels):
    pred = (soft_bits[..., 1] > 0.5).float()
    return (pred != labels).float().mean().item()
