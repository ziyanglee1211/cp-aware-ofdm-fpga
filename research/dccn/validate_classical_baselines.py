"""Numerical validation for the optimized CP-enhanced LS implementation."""

import torch

from benchmarks import ConventionalReceiver, ofdm_frontend
from config import OfdmConfig
from ofdm import ChannelTorch, OfdmTxTorch
from train import apply_fading_awgn


def direct_ls_cp(conv, Y, H_ls, y_cp):
    """Small-batch direct least-squares reference matching cpenhanced.m."""
    cfg = conv.cfg
    n, F, K = Y.shape
    L = cfg.CP
    H0 = H_ls.mean(dim=1, keepdim=True).expand(-1, F, -1)
    x_ls = torch.fft.ifft(Y / (H0 + 1e-9), dim=-1)
    rows = torch.arange(K - L, K, dtype=Y.real.dtype).view(L, 1)
    cols = torch.arange(K, dtype=Y.real.dtype).view(1, K)
    q_last = (torch.exp(2j * torch.pi * rows * cols / K) / K).to(Y.dtype)
    prev = torch.zeros(n, L, dtype=Y.dtype)
    output = []
    for sym in range(F):
        cur = x_ls[:, sym, -L:]
        xcp = conv._circshift_comb(prev, cur)
        h = (torch.linalg.pinv(xcp) @ y_cp[:, sym, :L].unsqueeze(-1)).squeeze(-1)
        xcp_u = torch.cat((torch.zeros(n, L, 1, dtype=Y.dtype),
                           torch.triu(xcp[:, :, 1:], diagonal=1)), dim=-1)
        c_cp = y_cp[:, sym, :L] - (xcp_u @ h.unsqueeze(-1)).squeeze(-1)
        h_l = conv._circshift_comb(torch.zeros_like(h), h)
        A = h_l @ q_last
        direct = []
        for item in range(n):
            B = torch.cat((torch.diag(H0[item, sym]), A[item]), dim=0)
            C = torch.cat((Y[item, sym], c_cp[item]), dim=0)
            direct.append(torch.linalg.lstsq(B, C).solution)
        output.append(torch.stack(direct))
        prev = cur
    return torch.stack(output, dim=1)


def main():
    torch.manual_seed(20260918)
    etu_db = torch.tensor([-1., -1., -1., 0., 0., 0., -3., -5., -7.])
    expected = torch.pow(10.0, etu_db / 10.0)
    expected /= expected.sum()
    for long_cp in (False, True):
        cfg = OfdmConfig(nbits=1, long_cp=long_cp)
        tx = OfdmTxTorch(cfg, torch.device("cpu"))
        ch = ChannelTorch(cfg, torch.device("cpu"), model="ETU")
        if not torch.allclose(ch.pdp, expected, atol=1e-7):
            raise SystemExit("ETU dB-to-linear PDP conversion is incorrect")
        conv = ConventionalReceiver(cfg, torch.device("cpu"))
        x, _, _ = tx.transmit(2)
        y, _ = apply_fading_awgn(ch, x, torch.full((2,), 16.0))
        Y = ofdm_frontend(y, cfg)
        H_ls = conv.ls_spline(Y)
        fast = conv.ls_cp_equalize(Y, H_ls, y)
        direct = direct_ls_cp(conv, Y, H_ls, y)
        max_abs = (fast - direct).abs().max().item()
        rel = ((fast - direct).abs().norm() / direct.abs().norm()).item()
        print(f"CP={cfg.CP}: max_abs={max_abs:.3e}, relative_l2={rel:.3e}")
        if rel > 2e-4:
            raise SystemExit("Woodbury and direct CP-enhanced LS disagree")
    print("PASS")


if __name__ == "__main__":
    main()
