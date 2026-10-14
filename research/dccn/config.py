"""DCCN OFDM system configuration (paper Table I, LTE-like downlink)."""

import numpy as np


class OfdmConfig:
    N = 64              # DFT size
    N_SYM = 7           # OFDM symbols per coherence slot (F)
    DC = 2              # DC guard subcarriers
    N_RB = 4            # LTE resource blocks -> P = 2*nRB pilots per pilot symbol
    LONG_CP = 16        # 25% of N
    SHORT_CP = 4        # ~7% of N

    PILOT_VALUE = 0.5 + 0.5j   # paper Sec. V-A: constant pilot 1/2(1+i)
    FS = 0.96e6               # sample rate

    def __init__(self, nbits=1, long_cp=True):
        self.nbits = nbits
        self.m = nbits
        self.K = self.N
        self.CP = self.LONG_CP if long_cp else self.SHORT_CP
        self.S = self.N + self.CP          # full time-domain symbol length
        self.P = 2 * self.N_RB             # pilots per pilot symbol = 8
        self.G = self.K - self.DC - self.N_RB * 12   # guard SCs = 14

        # effective subcarriers (excluding edge guard + DC)
        self.effec = np.arange(self.G // 2, self.K - self.G // 2)
        self.effec = np.setdiff1d(self.effec, np.arange(self.K // 2 - 1, self.K // 2 + 1))
        n_eff = len(self.effec)            # 48
        # LTE scattered pilots: symbol 0 and 4, 8 pilots each, freq shift of 3
        loc0 = np.arange(0, n_eff, int(np.ceil(n_eff / self.P)))  # [0,6,...,42]
        self.pilot_sc_sym = {
            0: self.effec[loc0],
            4: self.effec[(loc0 + 3) % n_eff],
        }
        pilot_set = set((s, sc) for s in (0, 4) for sc in self.pilot_sc_sym[s])

        # per-slot flat indices: (sym, sc) -> grid position
        self.pilot_pos = sorted(pilot_set)
        self.data_pos = sorted((s, sc) for s in range(self.N_SYM)
                               for sc in self.effec if (s, sc) not in pilot_set)
        self.P_slot = len(self.pilot_pos)  # 16
        self.D_slot = len(self.data_pos)   # 320

        # constellation (Gray coded, unit average power)
        self.const, self.bit2idx = make_constellation(nbits)

    def summary(self):
        return (f"N={self.K} F={self.N_SYM} CP={self.CP} S={self.S} "
                f"pilots/slot={self.P_slot} data/slot={self.D_slot} nbits={self.nbits}")


def make_constellation(nbits):
    """Gray-mapped constellation with unit average power. Returns (points, bit->index map)."""
    if nbits == 1:   # BPSK
        pts = np.array([+1.0 + 0j, -1.0 + 0j], dtype=np.complex64)
        bmap = {(0,): 0, (1,): 1}
    elif nbits == 2:  # QPSK Gray
        pts = np.array([+1 + 1j, +1 - 1j, -1 - 1j, -1 + 1j], dtype=np.complex64) / np.sqrt(2)
        bmap = {(0, 0): 0, (0, 1): 1, (1, 1): 2, (1, 0): 3}
    elif nbits == 4:  # 16QAM Gray
        levels = np.array([-3, -1, +1, +3]) / np.sqrt(10)
        gray = [3, 2, 0, 1]  # level index -> Gray code bits order (b1b0)
        pts, bmap = [], {}
        for i_re in range(4):
            for i_im in range(4):
                idx = i_re * 4 + i_im
                b = format(gray[i_re], '02b') + format(gray[i_im], '02b')
                pts.append(levels[i_re] + 1j * levels[i_im])
                bmap[tuple(int(c) for c in b)] = idx
        pts = np.array(pts, dtype=np.complex64)
    else:
        raise ValueError("only BPSK/QPSK/16QAM supported")
    pts = np.asarray(pts, dtype=np.complex64)   # keep complex64 after /sqrt() promotions
    return pts, bmap
