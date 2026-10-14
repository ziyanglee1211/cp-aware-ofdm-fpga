"""Fixed-point emulation operators for complex-valued quantization-aware
training of the DCCN receiver (paper plan C2).

Hardware contract emulated here:
  * weights: symmetric uniform, per-output-channel scale, re/im separate
  * activations: symmetric uniform, per-tensor STATIC scale (calibrated as
    running max over representative traffic, frozen at deployment)
  * complex MAC: re/im operands quantized to `abits`, accumulation exact
    (32-bit accumulator), output re-quantized to `abits`
  * per-slot RMS norm: reciprocal sqrt via integer power-of-2 shift + small
    LUT (values `lut_bits`); the scale is treated as constant (STE)
  * complex division X = Y conj(H)/(|H|^2 + delta):
      'lut_newton' : log-LUT reciprocal + one Newton step on the error term
                     (LUT 16 entries x 8-bit values; after refinement the
                     relative error is ~0.8% at 8-bit operands)
      'shift'      : r = 2^-ceil(log2 v), pure shift; r in (0.5, 1]/v so the
                     per-RE gain error reaches 2x but the PHASE is exact
      'float'      : oracle reference
  * tanh: piecewise-linear approximation, k uniform segments on [-R, R],
    slopes/intercepts in an 8-bit LUT
"""

import math
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F


def qmax_of(bits):
    return 2 ** (bits - 1) - 1


def fake_quant(x, scale, bits):
    """Symmetric uniform fake quantization with straight-through estimator."""
    q = (x / scale).round().clamp_(-qmax_of(bits), qmax_of(bits)) * scale
    return x + (q - x).detach()


@dataclass
class QConfig:
    wbits: int = 8
    abits: int = 8
    quant_w: bool = True          # weight quantization on/off
    quant_a: bool = True          # activation quantization on/off
    cmode: str = "cart"           # complex activation mode: 'cart' | 'polar'
    rsqrt: str = "lut"            # 'lut' | 'float'
    div: str = "lut_newton"       # 'lut_newton' | 'shift' | 'float'
    pwla_k: int = 16              # tanh PWLA segments; 0 = float tanh
    pwla_R: float = 3.0
    lut_entries: int = 16
    lut_bits: int = 8

    def name(self):
        n = f"w{self.wbits}a{self.abits}"
        if self.div != "lut_newton":
            n += f"_{self.div}"
        if self.pwla_k != 16:
            n += f"_pwla{self.pwla_k if self.pwla_k else 'off'}"
        if self.cmode != "cart":
            n += f"_{self.cmode}"
        if not self.quant_w:
            n += "_wF"
        if not self.quant_a:
            n += "_aF"
        return n


# ---------------------------------------------------------------- activations
class ActQ(nn.Module):
    """Real per-tensor activation quantizer with static scale.

    In training mode the running max keeps absorbing outliers (conservative);
    in eval mode the calibrated scale is frozen, as programmed in hardware."""

    def __init__(self, bits):
        super().__init__()
        self.bits = bits
        self.register_buffer("amax", torch.zeros(()))
        self.register_buffer("calibrated", torch.tensor(False))

    @torch.no_grad()
    def _observe(self, t):
        a = t.detach().abs().amax()
        if not bool(self.calibrated):
            self.amax.copy_(a)
            self.calibrated.fill_(True)
        elif a > self.amax:
            self.amax.copy_(a)

    def forward(self, t):
        if self.training:
            self._observe(t)
        if not bool(self.calibrated):
            return t
        scale = (self.amax * 1.02).clamp_min(1e-8) / qmax_of(self.bits)
        return fake_quant(t, scale, self.bits)


class ActQC(nn.Module):
    """Complex activation quantizer, Cartesian: re/im separate scales."""

    def __init__(self, qc: QConfig):
        super().__init__()
        self.qr = ActQ(qc.abits)
        self.qi = ActQ(qc.abits)

    def forward(self, z):
        return torch.complex(self.qr(z.real), self.qi(z.imag))


class ActQPolar(nn.Module):
    """Complex activation quantizer, polar: uniform magnitude + uniform phase.

    Included as an ablation baseline; the uniform magnitude grid is known to
    be inefficient for Rayleigh-distributed amplitudes."""

    def __init__(self, qc: QConfig):
        super().__init__()
        self.qr = ActQ(qc.abits)
        self.pbits = qc.abits

    def forward(self, z):
        r = z.abs()
        ph = z.angle()
        rq = self.qr(r)
        pq = fake_quant(ph, math.pi / qmax_of(self.pbits), self.pbits)
        return rq * torch.exp(1j * pq.to(rq.dtype))


def make_act_q(qc: QConfig):
    return ActQPolar(qc) if qc.cmode == "polar" else ActQC(qc)


# ------------------------------------------------------------------- weights
def _reduce_dims(W, axis):
    return tuple(i for i in range(W.dim()) if i != axis)


def quant_real_weight(W, bits, out_axis=0):
    dims = _reduce_dims(W, out_axis)
    s = W.abs().amax(dim=dims).clamp_min(1e-8) / qmax_of(bits)
    return fake_quant(W, s.view([-1 if i == out_axis else 1 for i in range(W.dim())]), bits)


def quant_complex_weight(W, bits, out_axis):
    """Complex weight quantization: per-output-channel scale, re/im separate."""
    dims = _reduce_dims(W, out_axis)
    shape = [-1 if i == out_axis else 1 for i in range(W.dim())]
    sr = W.real.abs().amax(dim=dims).clamp_min(1e-8) / qmax_of(bits)
    si = W.imag.abs().amax(dim=dims).clamp_min(1e-8) / qmax_of(bits)
    return torch.complex(fake_quant(W.real, sr.view(shape), bits),
                         fake_quant(W.imag, si.view(shape), bits))


# ---------------------------------------------------- fixed-point arithmetic
def _log_lut_recip(g, qc):
    """1/g for g in [2^-0.5, 2^0.5] via `lut_entries` uniform grid,
    values quantized to `lut_bits`."""
    lo, hi = 2 ** -0.5, 2 ** 0.5
    n = qc.lut_entries
    idx = ((g - lo) / (hi - lo) * n).clamp(0, n - 1).round().long()
    centers = lo + (idx.to(g.dtype) + 0.5) / n * (hi - lo)
    return fake_quant(centers.pow(-1), 1.5 / qmax_of(qc.lut_bits), qc.lut_bits)


def _log_lut_rsqrt(g, qc):
    """g^-0.5 for g in [2^-0.5, 2^0.5], values in [0.84, 1.19]."""
    lo, hi = 2 ** -0.5, 2 ** 0.5
    n = qc.lut_entries
    idx = ((g - lo) / (hi - lo) * n).clamp(0, n - 1).round().long()
    centers = lo + (idx.to(g.dtype) + 0.5) / n * (hi - lo)
    return fake_quant(centers.pow(-0.5), 1.2 / qmax_of(qc.lut_bits), qc.lut_bits)


def rsqrt_fixed(p, qc: QConfig):
    """p^-0.5 via integer shift + LUT.

    p = 2^(k+g) with k = round(log2 p): p^-0.5 = 2^(-k/2) * g^-0.5.
    The 2^(-k/2) factor is a power-of-2 shift when k is even and carries a
    sqrt(2) factor when odd (two-page LUT in hardware); numerically identical.
    """
    e = torch.log2(p.clamp_min(1e-12))
    k = torch.round(e)
    g = torch.exp2(e - k)
    return torch.exp2(-k / 2) * _log_lut_rsqrt(g, qc)


def recip_fixed(v, qc: QConfig):
    """1/v via log-LUT + one Newton step on the error term.

    r0 = 2^-k * LUT(g)          (rel. err ~ lut grid, <= ~3%)
    eps = v*r0 - 1  (quantized, small dynamic range)
    r  = r0 * (1 - eps)          (quantized operands)
    Newton turns the ~3% LUT error into ~0.1% at 8-bit operands.
    """
    e = torch.log2(v.clamp_min(1e-12))
    k = torch.round(e)
    g = torch.exp2(e - k)
    r0 = torch.exp2(-k) * _log_lut_recip(g, qc)
    eps = fake_quant(v * r0 - 1.0, 0.25 / qmax_of(qc.abits), qc.abits)
    u = fake_quant(1.0 - eps, 2.0 / qmax_of(qc.abits), qc.abits)
    return r0 * u


def recip_shift(v):
    """1/v via pure shift: r = 2^-ceil(log2 v) in (0.5, 1]/v."""
    return torch.exp2(-torch.ceil(torch.log2(v.clamp_min(1e-12))))


def rms_norm_fixed(x, qc: QConfig, eps=1e-8):
    """Per-slot RMS normalization with fixed-point reciprocal sqrt (STE)."""
    p = x.abs().square().mean(dim=(1, 2), keepdim=True)
    if qc.rsqrt == "float":
        s = (p + eps).sqrt().reciprocal()
    else:
        s = rsqrt_fixed(p, qc)
    return x * s.detach()


def cdiv_fixed(Y, H, delta, qc: QConfig):
    """X = Y conj(H) / (|H|^2 + delta), fixed-point emulation.

    v = |H|^2 + delta is accumulated exactly; the reciprocal is computed by
    the selected scheme; the final product is re-quantized by the caller.
    """
    v = H.abs().square() + delta
    if qc.div == "float":
        r = v.clamp_min(1e-12).reciprocal()
    elif qc.div == "shift":
        r = recip_shift(v)
    elif qc.div == "lut_newton":
        r = recip_fixed(v, qc)
    else:
        raise ValueError(qc.div)
    return Y * H.conj() * r


# ------------------------------------------------------------------ PWLA tanh
class PWLATanh(nn.Module):
    """Piecewise-linear tanh: k uniform segments on [-R, R].

    Outside [-R, R] the output saturates at +-tanh(R) (correct for tanh).
    Slopes/intercepts are quantized to 8-bit LUT entries. The backward pass
    is the PWLA subgradient (what the hardware computes), not the tanh one.
    """

    def __init__(self, k, R=3.0, bits=8):
        super().__init__()
        edges = torch.linspace(-R, R, k + 1)
        te = torch.tanh(edges)
        slopes = (te[1:] - te[:-1]) / (edges[1:] - edges[:-1])
        icepts = te[:-1] - slopes * edges[:-1]
        self.register_buffer("slopes", fake_quant(slopes, 1.0 / qmax_of(bits), bits))
        self.register_buffer("icepts", fake_quant(icepts, 1.0 / qmax_of(bits), bits))
        self.k, self.R = k, R

    def forward(self, x):
        idx = ((x + self.R) / (2 * self.R) * self.k).clamp(0, self.k - 1).long()
        return self.slopes[idx] * x + self.icepts[idx]

    def max_approx_err(self):
        xs = torch.linspace(-self.R, self.R, 20001)
        return (self(xs) - torch.tanh(xs)).abs().max().item()
