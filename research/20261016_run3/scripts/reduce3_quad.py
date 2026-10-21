"""Diagnostic (expensive): the three-input reduction (top three exact, other seven at b = -0.46) by dense quadrature.

Per ordered position triple, integrate over m2 (Gauss-Legendre on [-4.5, 4.5]), m1 > m2 via u = conditional CDF of m1 above m2,
and m3 < m2 via v = (Phi(m3)/Phi(m2))^8 (the weight of m3 times the seven inputs below it). Nodes in u and v are Gauss-Legendre
on (0, 1), which cluster at the ends where the near-tie bands sit. The value per model is the reduced accuracy without sampling;
it checks how far the cheap estimate (reduce3_det.py) is from the quantity it approximates. Saves per-triple, per-m2 results.
"""
import itertools
import sys

import numpy as np
import torch as th
from scipy.special import ndtr, ndtri

from algzoo.core import example, zoo

B = -0.46
N2, NU, NV = int(sys.argv[2]) if len(sys.argv) > 2 else 48, 32, 32
x2, w2 = np.polynomial.legendre.leggauss(N2); M2 = 4.5 * x2; W2 = 4.5 * w2
xu, wu = np.polynomial.legendre.leggauss(NU); U = (xu + 1) / 2; WU = wu / 2
xv, wv = np.polynomial.legendre.leggauss(NV); V = (xv + 1) / 2; WV = wv / 2


def quad(params, dev="cuda"):
    w_hi, w_hh, w_oh = (th.as_tensor(q, dtype=th.float64, device=dev) for q in params)
    T = np.array(list(itertools.permutations(range(10), 3)))
    P2 = ndtr(M2)
    m1 = ndtri(P2[:, None] + U[None, :] * (1 - P2[:, None]))            # [N2, NU]
    m3 = ndtri(V[None, :] ** (1 / 8) * P2[:, None])                       # [N2, NV]
    wt = np.exp(-M2 ** 2 / 2) / np.sqrt(2 * np.pi) * (1 - P2) * P2 ** 8 / 8 * W2   # weight per m2 node
    m1t, m3t = th.as_tensor(m1, device=dev), th.as_tensor(m3, device=dev)
    m2t = th.as_tensor(M2, device=dev)
    right = np.zeros((len(T), N2, NU, NV), dtype=bool)
    pred = np.zeros((len(T), N2, NU, NV), dtype=np.int8)
    for i, (a, b, c) in enumerate(T):
        X = th.full((N2, NU, NV, 10), B, dtype=th.float64, device=dev)
        X[..., a] = m1t[:, :, None]; X[..., b] = m2t[:, None, None]; X[..., c] = m3t[:, None, :]
        X = X.reshape(-1, 10)
        h = th.zeros(X.shape[0], len(w_hi), dtype=th.float64, device=dev)
        for t in range(10):
            h = th.relu(h @ w_hh.T + X[:, t, None] * w_hi)
        pr = (h @ w_oh.T).argmax(-1).reshape(N2, NU, NV).cpu().numpy()
        pred[i] = pr
        right[i] = pr == b
    per = (right * WU[None, None, :, None] * WV[None, None, None, :]).sum((2, 3))   # [720, N2] conditional right fraction
    I = 720 * (per * wt[None, :]).sum(1)
    return I.mean(), I, right, wt, pred


if __name__ == "__main__":
    name = sys.argv[1]
    p = example() if name == "example" else zoo(16, 10, int(name))
    est, I, right, wt, pred = quad(p)
    np.savez_compressed(f"/data/output/research/algzoo/20261016_run3/reports/reduce3_quad_{name}.npz", I=I, right=right, wt=wt, pred=pred)
    print(f"{name}: reduction by quadrature {est:.4f} (nodes {N2}x{NU}x{NV}, weight total {720 * wt.sum() / 720:.6f})")
