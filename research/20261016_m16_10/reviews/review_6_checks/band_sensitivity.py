"""Review 6, check B: how much do the frozen band estimate's free settings move its error on the six trained models?

The three-point band rule (reduce3_det.py) has settings that were set by hand: the level of the second used for the tie runs
(L_MID = 0.9), the off-tie step (EPS = 0.02), the bulk gaps (D12 = 0.5, D23 = 0.4), the bulk levels (0.2, 1.2) and the number
of Gauss-Legendre nodes in the final 1-D integral (64). If nearby reasonable settings give very different errors, the
reported error (RMSE 0.78 points) is partly luck of the chosen setting. Also reports what the integral costs with fewer nodes.
Scored only on ARC's model and zoo seeds 0-4. Truth values are the run's (reduce3_det.py).
"""
import itertools
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "2")
import numpy as np
from scipy.special import ndtr

sys.path.insert(0, "/data/output/research/algzoo/20261016_run3/scripts")
import reduce3_det as R  # noqa: E402

from algzoo.core import example, zoo  # noqa: E402

TRUTH = {"example": 0.9532, 0: 0.9101, 1: 0.8804, 2: 0.9485, 3: 0.9320, 4: 0.8898}
T = np.array(list(itertools.permutations(range(10), 3))); n = len(T); ar = np.arange(n)


def estimate(params, B=-0.46, D12=0.5, D23=0.4, L_LO=0.2, L_HI=1.2, L_MID=0.9, EPS=0.02, nodes=64):
    params = tuple(np.asarray(q, np.float64) for q in params)
    x, w = np.polynomial.legendre.leggauss(nodes); M2 = 4.5 * x; W2 = 4.5 * w
    PHI2 = np.exp(-M2 ** 2 / 2) / np.sqrt(2 * np.pi)

    def inputs(m1, m2, m3):
        X = np.full((n, 10), B); X[ar, T[:, 0]] = m1; X[ar, T[:, 1]] = m2; X[ar, T[:, 2]] = m3
        return X

    def marg(X):
        s, _ = R.run_batch(params, X)
        return R.margin(s, T[:, 1])[0]

    Mlo = marg(inputs(L_LO + D12, L_LO, L_LO - D23)); Mhi = marg(inputs(L_HI + D12, L_HI, L_HI - D23))
    Mmid = marg(inputs(L_MID + D12, L_MID, L_MID - D23))

    def band(M0, Me, Md, eps, D):
        w_ = np.zeros_like(M0)
        w_ = np.where((M0 < 0) & (Me >= 0), eps * (-M0) / np.maximum(Me - M0, 1e-12), w_)
        w_ = np.where((M0 < 0) & (Me < 0) & (Md > 0), eps + (D - eps) * (-Me) / np.maximum(Md - Me, 1e-12), w_)
        w_ = np.where((M0 < 0) & (Me < 0) & (Md <= 0), D, w_)
        return w_

    M23 = marg(inputs(L_MID + D12, L_MID, L_MID)); M23e = marg(inputs(L_MID + D12, L_MID, L_MID - EPS))
    w23 = band(M23, M23e, Mmid, EPS, D23)
    M12 = marg(inputs(L_MID, L_MID, L_MID - D23)); M12e = marg(inputs(L_MID + EPS, L_MID, L_MID - D23))
    w12 = band(M12, M12e, Mmid, EPS, D12)
    sc = np.maximum((M2[None, :] - B) / (L_MID - B), 0)
    Mm = Mlo[:, None] + (Mhi - Mlo)[:, None] * (M2[None, :] - L_LO) / (L_HI - L_LO)
    C = (Mm > 0).astype(float)
    f = 720 * PHI2[None, :] * C * (1 - ndtr(M2[None, :] + w12[:, None] * sc)) * ndtr(M2[None, :] - w23[:, None] * sc) ** 8 / 8
    return (f * W2[None, :]).sum(1).mean()


if __name__ == "__main__":
    models = {k: (example() if k == "example" else zoo(16, 10, k)) for k in TRUTH}
    settings = [dict()]
    settings += [dict(nodes=k) for k in (12, 16, 24, 32)]
    settings += [dict(L_MID=v) for v in (0.6, 0.75, 1.05, 1.2)]
    settings += [dict(EPS=v) for v in (0.01, 0.03, 0.04)]
    settings += [dict(D12=0.4, D23=0.3), dict(D12=0.6, D23=0.5), dict(L_LO=0.1, L_HI=1.0), dict(L_LO=0.3, L_HI=1.5)]
    for s in settings:
        errs = np.array([100 * (estimate(models[k], **s) - TRUTH[k]) for k in TRUTH])
        lab = ", ".join(f"{a}={b}" for a, b in s.items()) or "default (as frozen)"
        print(f"{lab:28s} errors " + " ".join(f"{e:+.2f}" for e in errs) + f"  RMSE {np.sqrt(np.mean(errs ** 2)):.2f}", flush=True)
