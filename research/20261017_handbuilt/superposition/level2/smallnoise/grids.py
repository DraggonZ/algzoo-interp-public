"""Quadrature grids over the three largest values. The network has no biases, so its answer is unchanged when the whole
list is scaled by a positive number, but flips character where the second-largest changes sign; the old grid (third on
Gauss-Legendre) puts that jump inside cells. 'split' centres the grid on the second-largest m2 and splits it at 0;
'gl' also uses Gauss-Legendre in s for the gaps a, b = R s^2."""
import numpy as np
from scipy.stats import norm
import sys
sys.path.insert(0, "/data/output/research/algzoo/20261017_handbuilt/hilton/propagate")
from full_mean import grid as grid_old


def _s(na, gl, R):
    if gl:
        x, w = np.polynomial.legendre.leggauss(na); s = 0.5 * (x + 1); ws = 0.5 * w
    else:
        s = (np.arange(na) + 0.5) / na; ws = np.full(na, 1 / na)
    return R * s ** 2, R * 2 * s * ws


def grid_split(nu, na, gl=False, R=5.0, lo=-3.0, hi=4.0, centre="m2"):
    x, w = np.polynomial.legendre.leggauss(nu // 2)
    u = np.r_[0.5 * (x + 1) * (0 - lo) + lo, 0.5 * (x + 1) * hi]; wu = np.r_[0.5 * (0 - lo) * w, 0.5 * hi * w]
    a, wa = _s(na, gl, R)
    U, A, B = np.meshgrid(u, a, a, indexing="ij"); WU, WA, WB = np.meshgrid(wu, wa, wa, indexing="ij")
    if centre == "m2": m2 = U.ravel(); m3 = m2 - A.ravel(); m1 = m2 + B.ravel()
    else: m3 = U.ravel(); m2 = m3 + A.ravel(); m1 = m2 + B.ravel()
    wt = WU.ravel() * WA.ravel() * WB.ravel() * norm.pdf(m1) * norm.pdf(m2) * norm.pdf(m3) * norm.cdf(m3) ** 7
    return np.c_[m1, m2, m3], wt


def get(kind, nu, na):
    if kind == "old": return grid_old(nu, na)
    if kind == "split": return grid_split(nu, na)
    if kind == "split-gl": return grid_split(nu, na, gl=True)
    if kind == "split3-gl": return grid_split(nu, na, gl=True, lo=-3.5, hi=3.5, centre="m3")   # third on the grid, split at 0
