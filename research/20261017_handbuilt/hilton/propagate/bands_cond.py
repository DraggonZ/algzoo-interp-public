"""Copy of run 3's reduce3_det.py (band estimator) with one change: the seven smaller numbers at the mean of a normal truncated below the third-largest (COND = True) instead of the fixed -0.46."""
"""Sampling-free estimate of the three-input reduction (top three inputs exact, the other seven at b = -0.46).

Per ordered position triple (largest p1, second p2, third p3), with m2 the second's value:
  bulk:   margin (score of p2 minus best other) away from ties, at two levels of m2 -> threshold below which a small
          second fails (or the whole triple fails, or never fails);
  band23: width of the band m2 - m3 < w23 where the network is wrong, from the margin at the tie m3 = m2 and a straight line
          to the margin away from the tie, at one level, scaled with (m2 - b);
  band12: the same for m1 - m2 < w12.
Then I(triple) = int dm2 720 phi(m2) C(m2) (1 - Phi(m2 + w12(m2))) Phi(m2 - w23(m2))^8 / 8, the exact weight of the top-three
order statistics outside the bands (the order-statistic factors do not depend on the model). Estimate = mean over the
720 triples. Cost counts the network runs and the order-statistic evaluations.
"""
import itertools
import sys

import numpy as np
from scipy.special import ndtr

B = -0.46
COND = True
from scipy.stats import norm as _n
def BG(m3):
    """mean of a standard normal truncated below m3"""
    return -_n.pdf(m3) / _n.cdf(m3)
D12, D23 = 0.5, 0.4
L_LO, L_HI, L_MID = 0.2, 1.2, 0.9
_x, _w = np.polynomial.legendre.leggauss(64)
M2 = 4.5 * _x            # m2 nodes on [-4.5, 4.5]
W2 = 4.5 * _w
PHI2 = np.exp(-M2 ** 2 / 2) / np.sqrt(2 * np.pi)


def run_batch(params, X, dX=None):
    """Scores and (optionally) their derivative along dX for a batch of inputs."""
    w_hi, w_hh, w_oh = params
    h = np.zeros((X.shape[0], len(w_hi)))
    dh = np.zeros_like(h) if dX is not None else None
    for t in range(10):
        z = h @ w_hh.T + X[:, t, None] * w_hi
        on = (z > 0).astype(float)
        if dX is not None:
            dz = dh @ w_hh.T + dX[:, t, None] * w_hi
            dh = dz * on
        h = z * on
    s = h @ w_oh.T
    return s, (dh @ w_oh.T if dX is not None else None)


def margin(s, p2):
    o = s.copy(); o[np.arange(len(s)), p2] = -np.inf
    k = o.argmax(1)
    return s[np.arange(len(s)), p2] - s[np.arange(len(s)), k], k


def estimate(params, scale_widths=True):
    params = tuple(np.asarray(q, np.float64) for q in params)
    d = len(params[0])
    T = np.array(list(itertools.permutations(range(10), 3)))      # [720, 3]: p1, p2, p3
    n = len(T); ar = np.arange(n)
    ops = 0
    run_cost = 10 * (d * d + d) + params[2].size

    def inputs(m1, m2, m3):
        X = np.full((n, 10), BG(m3) if COND else B)
        X[ar, T[:, 0]] = m1; X[ar, T[:, 1]] = m2; X[ar, T[:, 2]] = m3
        return X

    # bulk at two levels
    Mb = []
    for L in (L_LO, L_HI):
        s, _ = run_batch(params, inputs(L + D12, L, L - D23)); ops += n * run_cost
        Mb.append(margin(s, T[:, 1])[0])
    Mlo, Mhi = Mb
    # bulk at the middle level (for the secant band widths)
    s, _ = run_batch(params, inputs(L_MID + D12, L_MID, L_MID - D23))  # cost counted below, for the triples that use it
    Mmid = margin(s, T[:, 1])[0]
    def band(M0, Me, Md, eps, D):
        """Band width from margins at the tie (M0), just off it (Me, at eps) and in the bulk (Md, at D)."""
        w = np.where(M0 >= 0, 0.0, 0.0)
        w = np.where((M0 < 0) & (Me >= 0), eps * (-M0) / np.maximum(Me - M0, 1e-12), w)
        w = np.where((M0 < 0) & (Me < 0) & (Md > 0), eps + (D - eps) * (-Me) / np.maximum(Md - Me, 1e-12), w)
        w = np.where((M0 < 0) & (Me < 0) & (Md <= 0), D, w)
        return w

    EPS = 0.02
    # band 23: margins at m3 = m2 and at m3 = m2 - EPS
    s, _ = run_batch(params, inputs(L_MID + D12, L_MID, L_MID)); ops += n * run_cost
    M23 = margin(s, T[:, 1])[0]
    s, _ = run_batch(params, inputs(L_MID + D12, L_MID, L_MID - EPS)); ops += int((M23 < 0).sum()) * run_cost
    M23e = margin(s, T[:, 1])[0]
    w23 = band(M23, M23e, Mmid, EPS, D23)
    # band 12: margins at m1 = m2 and at m1 = m2 + EPS
    s, _ = run_batch(params, inputs(L_MID, L_MID, L_MID - D23)); ops += n * run_cost
    M12 = margin(s, T[:, 1])[0]
    s, _ = run_batch(params, inputs(L_MID + EPS, L_MID, L_MID - D23)); ops += int((M12 < 0).sum()) * run_cost
    M12e = margin(s, T[:, 1])[0]
    w12 = band(M12, M12e, Mmid, EPS, D12)
    ops += int((((M23 < 0) & (M23e < 0)) | ((M12 < 0) & (M12e < 0))).sum()) * run_cost
    # small-second threshold from the bulk margins (linear in m2 between the two levels)
    sc = (M2[None, :] - B) / (L_MID - B) if scale_widths else np.ones((1, len(M2)))
    sc = np.maximum(sc, 0)
    Mm = Mlo[:, None] + (Mhi - Mlo)[:, None] * (M2[None, :] - L_LO) / (L_HI - L_LO)
    C = (Mm > 0).astype(float)
    W23 = w23[:, None] * sc; W12 = w12[:, None] * sc
    f = 720 * PHI2[None, :] * C * (1 - ndtr(M2[None, :] + W12)) * ndtr(M2[None, :] - W23) ** 8 / 8
    I = (f * W2[None, :]).sum(1)
    ops += n * len(M2) * 6
    return I.mean(), ops, dict(w23=w23, w12=w12, Mlo=Mlo, Mhi=Mhi)


