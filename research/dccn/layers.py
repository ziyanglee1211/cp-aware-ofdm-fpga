"""Complex-valued linear layers for PyTorch (native complex64 tensors + autograd)."""

import torch
import torch.nn as nn


class ComplexLinear(nn.Module):
    """Dense layer C^{in x out}: x[..., in] @ W^T + b with complex weights."""

    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        std = (1.0 / in_features) ** 0.5
        re = torch.randn(out_features, in_features) * (std / 2 ** 0.5)
        im = torch.randn(out_features, in_features) * (std / 2 ** 0.5)
        self.weight = nn.Parameter(torch.complex(re, im))
        self.bias = (nn.Parameter(torch.zeros(out_features, dtype=torch.complex64))
                     if bias else None)

    def forward(self, x):
        x = x @ self.weight.T.to(x.dtype)
        if self.bias is not None:
            x = x + self.bias.to(x.dtype)
        return x


class ComplexLinearSN(nn.Module):
    """Complex linear transform shared across OFDM symbols.

    Implements the DFT-like / IDFT-like C-Conv layers of the paper:
    y[b, f, n] = sum_s x[b, f, s] * W[s, n] + b[n]  (1D complex conv, N filters).
    """

    def __init__(self, s_len, n_len, bias=True):
        super().__init__()
        std = (1.0 / s_len) ** 0.5
        re = torch.randn(s_len, n_len) * (std / 2 ** 0.5)
        im = torch.randn(s_len, n_len) * (std / 2 ** 0.5)
        self.weight = nn.Parameter(torch.complex(re, im))
        self.bias = (nn.Parameter(torch.zeros(n_len, dtype=torch.complex64))
                     if bias else None)

    def forward(self, x):
        y = torch.einsum('bfs,sn->bfn', x, self.weight.to(x.dtype))
        if self.bias is not None:
            y = y + self.bias.to(x.dtype)
        return y


def rms_norm(x, eps=1e-8):
    """Per-slot RMS normalization (real positive scaling, phase preserved)."""
    p = x.abs().square().mean(dim=(1, 2), keepdim=True)
    return x / (p.sqrt() + eps)
