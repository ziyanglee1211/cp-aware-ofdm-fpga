"""Conventional OFDM receiver benchmarks.

The implementations follow ``OFDM_Benchmark_dev.m`` from the public
Deep-Waveform repository:

* LS-Spline + ZF (``eq_idx=2``)
* ideal LMMSE using the true channel covariance (``eq_idx=4``)
* approximate LMMSE estimated from the LS response (``eq_idx=7``)
* cyclic-prefix-enhanced LS equalization (``eq_idx=5``)

The CP-enhanced solver is algebraically equivalent to the MATLAB normal
equations, but uses the Woodbury identity.  This reduces the per-symbol solve
from 64 x 64 to CP x CP (4 x 4 or 16 x 16 in the evaluated systems).
"""

import numpy as np
import torch

from config import OfdmConfig


class ConventionalReceiver:
    def __init__(self, cfg: OfdmConfig, device):
        self.cfg = cfg
        self.device = device
        # pilot value and positions
        self.pv = cfg.PILOT_VALUE
        self.pilot_sc_sym = cfg.pilot_sc_sym
        # data RE positions (sym, sc) for decision comparison
        self.data_pos = cfg.data_pos

    def ls_spline(self, Y):
        """LS pilot estimates + cubic spline (freq) + linear (time) interpolation.

        Y: [n, F, K] frequency-domain received grid (after exact DFT + CP removal).
        Returns H_hat [n, F, K].
        """
        cfg = self.cfg
        n = Y.shape[0]
        Y = Y.detach().cpu().numpy()
        H_hat = np.zeros((n, cfg.N_SYM, cfg.K), dtype=np.complex64)
        for s in (0, 4):
            sc = self.pilot_sc_sym[s]
            h_ls = Y[:, s, sc] / self.pv                    # [n, P]
            # clamp outside the pilot span (cubic extrapolation of noisy pilots
            # diverges; the reference 'v4' biharmonic spline extends smoothly)
            kk = np.clip(np.arange(cfg.K), sc[0], sc[-1])
            H_hat[:, s, :] = self._not_a_knot_spline(sc, h_ls, kk)
        # linear interpolation / extrapolation in time between sym 0 and 4
        for s in range(1, 4):
            w = s / 4.0
            H_hat[:, s, :] = (1 - w) * H_hat[:, 0, :] + w * H_hat[:, 4, :]
        for s in range(5, cfg.N_SYM):
            H_hat[:, s, :] = H_hat[:, 4, :]
        return torch.from_numpy(H_hat).to(self.device)

    @staticmethod
    def _not_a_knot_spline(x, y, query):
        """Pure-NumPy not-a-knot cubic spline with clamped query points.

        This matches ``scipy.interpolate.CubicSpline``'s default boundary
        condition for the eight pilot points, while avoiding a SciPy runtime
        dependency in the historical PyTorch environment.
        ``y`` has shape ``[batch, len(x)]``.
        """
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y)
        q = np.asarray(query, dtype=np.float64)
        count = len(x)
        if count < 4:
            raise ValueError("not-a-knot spline requires at least four points")
        h = np.diff(x)
        A = np.zeros((count, count), dtype=np.float64)
        rhs = np.zeros((y.shape[0], count), dtype=y.dtype)

        # Equality of the third derivative across the first and last knots.
        A[0, 0] = -h[1]
        A[0, 1] = h[0] + h[1]
        A[0, 2] = -h[0]
        A[-1, -3] = h[-1]
        A[-1, -2] = -(h[-2] + h[-1])
        A[-1, -1] = h[-2]
        for i in range(1, count - 1):
            A[i, i - 1] = h[i - 1]
            A[i, i] = 2.0 * (h[i - 1] + h[i])
            A[i, i + 1] = h[i]
            rhs[:, i] = 6.0 * ((y[:, i + 1] - y[:, i]) / h[i]
                               - (y[:, i] - y[:, i - 1]) / h[i - 1])
        second = np.linalg.solve(A, rhs.T).T

        segment = np.searchsorted(x, q, side="right") - 1
        segment = np.clip(segment, 0, count - 2)
        x0 = x[segment]
        x1 = x[segment + 1]
        width = x1 - x0
        left = (x1 - q) / width
        right = (q - x0) / width
        m0 = second[:, segment]
        m1 = second[:, segment + 1]
        y0 = y[:, segment]
        y1 = y[:, segment + 1]
        return (m0 * ((x1 - q) ** 3) / (6.0 * width)
                + m1 * ((q - x0) ** 3) / (6.0 * width)
                + (y0 - m0 * width ** 2 / 6.0) * left
                + (y1 - m1 * width ** 2 / 6.0) * right)

    def ideal_lmmse(self, H_ls, H_true, snr_db):
        """Oracle-assisted rank-one LMMSE estimator from the reference benchmark
        (OFDM_Benchmark_dev.m, eq_idx=4): per OFDM symbol the covariance is
        the rank-1 outer product of the TRUE channel response,
            W = Rhh (Rhh + (beta/snr) I)^{-1},  Rhh = h h^H,
        applied to the spline-interpolated LS estimate. Sherman-Morrison
        reduces this to a projection onto the true-channel direction:
            H_lmmse = h * (h^H H_ls) / (beta*sigma2 + |h|^2),
        with sigma2 = per-RE noise variance and beta = E|X|^4 / (E|X|^2)^2
        (=1 for BPSK/QPSK, 17/9 for 16QAM, as in the reference betas table).
        """
        sigma2 = 10 ** (-snr_db / 10.0)
        const = torch.from_numpy(self.cfg.const)
        beta = ((const.abs() ** 4).mean() / (const.abs() ** 2).mean() ** 2).item()
        alpha = beta * sigma2
        num = (H_true.conj() * H_ls).sum(dim=-1, keepdim=True)      # [n, F, 1]
        den = alpha + H_true.abs().square().sum(dim=-1, keepdim=True)
        return H_true * (num / den)

    def approx_lmmse(self, H_ls, snr_db):
        """Approximate LMMSE estimator from the reference ``eq_idx=7``.

        The two-dimensional LS estimate is averaged over the seven OFDM
        symbols.  Its rank-one covariance ``h h^H / F`` is then used in the
        LMMSE filter and the resulting response is repeated over the slot.
        Sherman-Morrison gives the scalar shrinkage used below.
        """
        sigma2 = 10 ** (-snr_db / 10.0)
        const = torch.from_numpy(self.cfg.const).to(H_ls.device)
        beta = ((const.abs() ** 4).mean()
                / (const.abs() ** 2).mean().square()).item()
        h = H_ls.mean(dim=1)                                      # [n, K]
        energy = h.abs().square().sum(dim=-1, keepdim=True)       # [n, 1]
        gain = energy / (energy + beta * sigma2 * self.cfg.N_SYM)
        return (h * gain).unsqueeze(1).expand(-1, self.cfg.N_SYM, -1)

    @staticmethod
    def _circshift_comb(vec_u, vec_l):
        """Vectorized port of ``cpenhanced.m::circshift_comb``.

        ``vec_u`` and ``vec_l`` have shape ``[batch, L]``.  The result has
        shape ``[batch, L, L]``.
        """
        L = vec_l.shape[-1]
        base = torch.cat((vec_u, vec_l), dim=-1)                  # [b, 2L]
        rows = torch.arange(L, 2 * L, device=base.device).view(L, 1)
        cols = torch.arange(L, device=base.device).view(1, L)
        indices = (rows - cols) % (2 * L)                         # [L, L]
        return base[:, indices]

    def ls_cp_equalize(self, Y, H_ls, y_cp, eps=1e-9):
        """CP-enhanced LS equalizer from ``cpenhanced.m`` (``eq_idx=5``).

        Parameters
        ----------
        Y:
            Received frequency-domain grid ``[n, F, K]``.
        H_ls:
            Interpolated LS response ``[n, F, K]``.
        y_cp:
            Received time-domain symbols including CP ``[n, F, K+CP]``.

        Returns
        -------
        Complex equalized grid ``[n, F, K]`` ready for hard detection.
        """
        cfg = self.cfg
        n, F, K = Y.shape
        L = cfg.CP
        if K != cfg.K or F != cfg.N_SYM or y_cp.shape[-1] != K + L:
            raise ValueError((Y.shape, H_ls.shape, y_cp.shape, K, L))

        # Reference eq_idx=5 averages the interpolated LS response in time.
        H0 = H_ls.mean(dim=1, keepdim=True).expand(-1, F, -1)
        X_ls = Y / (H0 + eps)
        x_ls = torch.fft.ifft(X_ls, dim=-1)
        y_prefix = y_cp[..., :L]

        # Last L rows of the inverse DFT matrix used by the MATLAB code.
        time_idx = torch.arange(K - L, K, device=Y.device,
                                dtype=Y.real.dtype).view(L, 1)
        freq_idx = torch.arange(K, device=Y.device,
                                dtype=Y.real.dtype).view(1, K)
        q_last = torch.exp(2j * torch.pi * time_idx * freq_idx / K) / K
        q_last = q_last.to(Y.dtype)                                # [L, K]

        prev = torch.zeros(n, L, dtype=Y.dtype, device=Y.device)
        out = []
        eye = torch.eye(L, dtype=Y.dtype, device=Y.device).unsqueeze(0)
        for sym in range(F):
            cur = x_ls[:, sym, -L:]
            xcp = self._circshift_comb(prev, cur)                  # [n,L,L]
            # The reference uses pinv when the normal equations are ill
            # conditioned.  pinv(xcp) is the numerically safer equivalent.
            h = torch.matmul(torch.linalg.pinv(xcp),
                             y_prefix[:, sym].unsqueeze(-1)).squeeze(-1)

            xcp_shift = xcp[:, :, 1:]
            xcp_u = torch.cat((torch.zeros(n, L, 1, dtype=Y.dtype,
                                           device=Y.device),
                               torch.triu(xcp_shift, diagonal=1)), dim=-1)
            c_cp = y_prefix[:, sym] - torch.matmul(
                xcp_u, h.unsqueeze(-1)).squeeze(-1)

            h_l = self._circshift_comb(torch.zeros_like(h), h)
            A = torch.matmul(h_l, q_last)                           # [n,L,K]

            # Solve min ||diag(H0)x-Y||^2 + ||Ax-c_cp||^2.
            # Woodbury form: x=x0+K^-1 A^H(I+A K^-1 A^H)^-1(c-Ax0).
            hs = H0[:, sym]
            inv_power = 1.0 / (hs.abs().square() + eps)            # [n,K]
            x0 = Y[:, sym] / (hs + eps)
            residual = c_cp - torch.matmul(A, x0.unsqueeze(-1)).squeeze(-1)
            weighted_A = A * inv_power.unsqueeze(1)
            small = eye + torch.matmul(weighted_A, A.conj().transpose(-2, -1))
            z = torch.linalg.solve(small, residual.unsqueeze(-1))
            correction = inv_power * torch.matmul(
                A.conj().transpose(-2, -1), z).squeeze(-1)
            out.append(x0 + correction)
            prev = cur

        return torch.stack(out, dim=1)

    def detect(self, Y, H_hat):
        """ZF equalization + hard decision on data REs. Returns bits_pm1 [n, D, m]."""
        cfg = self.cfg
        X = Y / (H_hat + 1e-12)
        n = Y.shape[0]
        idx = torch.tensor([s * cfg.K + sc for s, sc in self.data_pos],
                           device=Y.device).long()
        xd = X.reshape(n, -1)[:, idx]                       # [n, D] complex
        # minimum-distance hard decision over the constellation
        const = torch.from_numpy(cfg.const).to(Y.device)    # [M]
        bits_lut = torch.zeros(len(const), cfg.m, device=Y.device)
        for bits, ci in cfg.bit2idx.items():
            bits_lut[ci] = torch.tensor(bits, device=Y.device, dtype=torch.float32)
        d = (xd.unsqueeze(-1) - const.view(1, 1, -1)).abs().square()  # [n, D, M]
        hard = d.argmin(dim=-1)                              # [n, D]
        return bits_lut[hard] * 2 - 1

    def detect_equalized(self, X):
        """Hard decision for an already equalized frequency-domain grid."""
        cfg = self.cfg
        n = X.shape[0]
        idx = torch.tensor([s * cfg.K + sc for s, sc in self.data_pos],
                           device=X.device).long()
        xd = X.reshape(n, -1)[:, idx]
        const = torch.from_numpy(cfg.const).to(X.device)
        bits_lut = torch.zeros(len(const), cfg.m, device=X.device)
        for bits, ci in cfg.bit2idx.items():
            bits_lut[ci] = torch.tensor(bits, device=X.device,
                                        dtype=torch.float32)
        distances = (xd.unsqueeze(-1) - const.view(1, 1, -1)).abs().square()
        return bits_lut[distances.argmin(dim=-1)] * 2 - 1

    def ber(self, Y, H_hat, labels_pm1, snr_db=None, H_true=None):
        """BER of LS-Spline(+LMMSE) + ZF pipeline. labels_pm1: [n, D, m] +-1."""
        if H_true is not None:
            H_hat = self.ideal_lmmse(H_hat, H_true, snr_db)
        bits = self.detect(Y, H_hat)
        return (bits != labels_pm1).float().mean().item()


def ofdm_frontend(y_cp, cfg: OfdmConfig):
    """Conventional frontend: remove CP + unitary DFT. y_cp [n,F,S] -> Y [n,F,K]."""
    x = y_cp[..., cfg.CP:]
    return torch.fft.fft(x, dim=-1) / (cfg.K ** 0.5)
