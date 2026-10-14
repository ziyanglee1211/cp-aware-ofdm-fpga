"""OFDM transmitter + channel, fully on torch (GPU-friendly online data generation)."""

import os
import numpy as np
import torch

from config import OfdmConfig


class OfdmTxTorch:
    """Generates OFDM coherence slots from random bits, on device."""

    def __init__(self, cfg: OfdmConfig, device):
        self.cfg = cfg
        self.device = device
        c = torch.from_numpy(cfg.const)  # [M]
        self.const = c.to(device)
        # data positions as (sym_idx * N + sc) flat grid indices
        self.data_flat = torch.tensor([s * cfg.K + sc for s, sc in cfg.data_pos],
                                      dtype=torch.long, device=device)
        self.pilot_flat = torch.tensor([s * cfg.K + sc for s, sc in cfg.pilot_pos],
                                       dtype=torch.long, device=device)
        # bit patterns for each constellation index
        bits_lut = torch.zeros(len(cfg.const), cfg.m, dtype=torch.float32)
        for bits, idx in cfg.bit2idx.items():
            bits_lut[idx] = torch.tensor(bits)
        self.bits_lut = bits_lut.to(device)

    def transmit(self, n_slots):
        """Returns (y_cp [n,F,S] complex, labels [n,D,m] float 0/1, bits_pm1 [n,D,m])."""
        cfg = self.cfg
        n, D, m = n_slots, cfg.D_slot, cfg.m
        # random constellation symbols for data REs
        idx = torch.randint(0, len(cfg.const), (n, D), device=self.device)
        syms = self.const[idx]                          # [n, D]
        labels = self.bits_lut[idx]                     # [n, D, m] in {0,1}
        # build frequency-domain grid [n, F*N]
        grid = torch.zeros(n, cfg.N_SYM * cfg.K, dtype=torch.complex64, device=self.device)
        grid.scatter_(1, self.data_flat.unsqueeze(0).expand(n, -1), syms)
        grid.scatter_(1, self.pilot_flat.unsqueeze(0).expand(n, -1),
                      torch.full((n, cfg.P_slot), cfg.PILOT_VALUE,
                                 dtype=torch.complex64, device=self.device))
        grid = grid.view(n, cfg.N_SYM, cfg.K)
        # unitary IDFT (x = sqrt(N) * ifft) so that per-sample time-domain power
        # equals per-RE frequency-domain power; noise sigma2 is then exactly
        # the per-data-RE noise variance (Es/N0 basis)
        x = torch.fft.ifft(grid, dim=-1) * (cfg.K ** 0.5)
        cp = x[..., -cfg.CP:]
        x_cp = torch.cat([cp, x], dim=-1)               # [n, F, S]
        bits_pm1 = labels * 2.0 - 1.0
        return x_cp, labels, bits_pm1


class ChannelTorch:
    """Static (slot-invariant) Rayleigh fading + AWGN.

    Fading taps follow the 3GPP EPA/EVA/ETU PDPs resampled to Fs with the
    same sinc resampling matrices shipped with the original repo (dev/py/3gpp/*.csv),
    so channel statistics match the paper.
    """

    def __init__(self, cfg: OfdmConfig, device, model="flat", csv_dir=None,
                 legacy_db_normalization=False):
        self.cfg = cfg
        self.device = device
        self.model = model
        self.legacy_db_normalization = legacy_db_normalization
        if csv_dir is None:
            # Keep experiments portable across machines.  The original code
            # used an absolute path from the author's workstation.
            csv_dir = os.path.normpath(os.path.join(
                os.path.dirname(__file__), "..", "code", "dl_ofdm-master",
                "dev", "py", "3gpp"))
        if model == "flat":
            self.L = 1
            self.A = torch.ones(1, 1, device=device)
            self.pdp = torch.ones(1, device=device)
            self.pdp_db = torch.zeros(1, device=device)
            self.pdp_mode = "flat"
        else:
            fname = {"EPA": "AM_EPA.csv", "EVA": "AM_EVA.csv", "ETU": "AM_ETU.csv"}[model]
            A = np.genfromtxt(csv_dir + "\\" + fname, delimiter=",")  # [K_paths, L]
            # PDP exactly as configured in the repo's MATLAB benchmark
            # (OFDM_Benchmark_dev.m lines 177-186); EVA/ETU use 9 paths
            pdp_db = {"EPA": [0, -1, -2, -3, -8, -17.2, -20.8],
                      "EVA": [0.0, -1.5, -1.4, -3.6, -0.6, -9.1, -7.0, -12.0, -16.9],
                      "ETU": [-1, -1, -1, 0, 0, 0, -3, -5, -7]}[model]
            assert A.shape[0] == len(pdp_db), (A.shape, model)
            self.A = torch.from_numpy(A).float().to(device)          # [K_paths, L]
            self.pdp_db = torch.tensor(pdp_db, device=device).float()
            if legacy_db_normalization:
                # Compatibility only for checkpoints/results produced before
                # 2026-09-18.  This historical path is not physically valid:
                # dB values must be converted to linear power before use.
                self.pdp = self.pdp_db / self.pdp_db.sum()
                self.pdp_mode = "legacy_direct_db_normalization"
            else:
                self.pdp = torch.pow(10.0, self.pdp_db / 10.0)
                self.pdp = self.pdp / self.pdp.sum()
                self.pdp_mode = "3gpp_db_to_linear_power"
            self.L = A.shape[1]

    def sample_h(self, n):
        """Complex FIR taps [n, L], true Rayleigh statistics: per-path
        CN(0, pdp) with sum(pdp)=1, so E[|h|^2]=1 (average SNR basis,
        matching the reference rayleighchan + awgn('measured') setup)."""
        z = (torch.randn(n, len(self.pdp), 1, device=self.device)
             + 1j * torch.randn(n, len(self.pdp), 1, device=self.device)) \
            * (self.pdp.sqrt() / 1.41421356).view(1, -1, 1).to(torch.complex64)
        return (z * self.A.unsqueeze(0)).sum(dim=1)                  # [n, L]

    def apply(self, x_cp, snr_db, h=None):
        """x_cp [n,F,S] -> y [n,F,S] with per-slot static fading + AWGN.

        snr_db is Es/N0 per data RE (data symbol power = 1).
        """
        cfg = self.cfg
        n = x_cp.shape[0]
        if h is None:
            h = self.sample_h(n)
        x = x_cp.reshape(n, -1)                                      # [n, F*S]
        if self.L == 1:
            y = x * h
        else:
            # frequency-domain fast convolution
            Lx = x.shape[1]
            nfft = 1 << (Lx + self.L - 2).bit_length()
            X = torch.fft.fft(x, n=nfft, dim=1)
            H = torch.fft.fft(h, n=nfft, dim=1)
            y = torch.fft.ifft(X * H, dim=1)[:, :Lx]
        sigma2 = 10 ** (-snr_db / 10.0)
        noise = (torch.randn_like(y) + 1j * torch.randn_like(y)) * ((sigma2 / 2) ** 0.5)
        y = (y + noise).reshape(n, cfg.N_SYM, cfg.S)
        # also produce true per-slot frequency response on all SCs (for benchmarks)
        Hf = torch.fft.fft(h, n=cfg.K, dim=1)                        # [n, K]
        return y.to(torch.complex64), Hf
