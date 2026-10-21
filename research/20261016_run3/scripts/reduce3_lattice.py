"""Gray-zone test: integrate the three-input reduction with a fixed lattice instead of random inputs.

Each ordered position triple has probability exactly 1/720. Within a triple, the top three values have the joint law
t1 = U1^(1/10), t2 = t1 U2^(1/9), t3 = t2 U3^(1/8) (uniform order statistics), m = Phi^-1(t), for U uniform on the cube.
We put n points of a rank-1 lattice in the cube (shifted per triple by a Kronecker offset, all deterministic), set the
other seven inputs to their conditional mean below m3, -phi(m3)/Phi(m3), and average "right". Cost 720 n runs.
This uses chosen inputs and numerical integration, not a deductive argument; it shows what the reduction is worth
when integrated well, and whether the reduced route can beat sampling at all.
"""
import itertools
import sys

import numpy as np
import torch as th
from scipy.special import ndtri
from scipy.stats import norm

T = np.array(list(itertools.permutations(range(10), 3)))
GEN = {1: (1, 1, 1), 2: (1, 1, 1), 4: (1, 1, 3), 8: (1, 3, 5), 16: (1, 5, 9), 32: (1, 9, 13), 64: (1, 19, 41)}
ALPHA = np.array([np.sqrt(2), np.sqrt(3), np.sqrt(5)]) % 1


def points(n):
    g = np.array(GEN[n]); i = np.arange(n)[:, None]
    base = (i * g[None, :] / n) % 1                               # [n, 3]
    shift = (np.arange(len(T))[:, None] * ALPHA[None, :]) % 1        # [720, 3]
    return (base[None, :, :] + shift[:, None, :]) % 1                 # [720, n, 3]


def estimate(params, n, dev="cuda"):
    U = np.clip(points(n), 1e-12, 1 - 1e-12)
    t1 = U[..., 0] ** (1 / 10); t2 = t1 * U[..., 1] ** (1 / 9); t3 = t2 * U[..., 2] ** (1 / 8)
    m1, m2, m3 = ndtri(t1), ndtri(t2), ndtri(t3)
    b = -norm.pdf(m3) / t3
    X = np.repeat(b[..., None], 10, -1)
    X[np.arange(len(T))[:, None], np.arange(n)[None, :], T[:, 0][:, None]] = m1
    X[np.arange(len(T))[:, None], np.arange(n)[None, :], T[:, 1][:, None]] = m2
    X[np.arange(len(T))[:, None], np.arange(n)[None, :], T[:, 2][:, None]] = m3
    w_hi, w_hh, w_oh = (th.as_tensor(q, dtype=th.float64, device=dev) for q in params)
    x = th.as_tensor(X.reshape(-1, 10), device=dev)
    h = th.zeros(x.shape[0], len(w_hi), dtype=th.float64, device=dev)
    for t in range(10):
        h = th.relu(h @ w_hh.T + x[:, t, None] * w_hi)
    right = ((h @ w_oh.T).argmax(-1).cpu().numpy().reshape(len(T), n) == T[:, 1:2])
    return right.mean()


if __name__ == "__main__":
    from algzoo.core import example, zoo
    truth = {"example": 0.9532, 0: 0.9101, 1: 0.8804, 2: 0.9485, 3: 0.9320, 4: 0.8898}
    for n in [1, 2, 4, 8, 16, 64]:
        errs = []
        for name in ["example", 0, 1, 2, 3, 4]:
            p = example() if name == "example" else zoo(16, 10, name)
            errs.append(100 * (estimate(p, n) - truth[name]))
        se = 100 * np.sqrt(np.mean([t * (1 - t) for t in truth.values()]) / (720 * n))
        print(f"n={n} per triple (cost {720 * n} runs): errors " + ", ".join(f"{e:+.2f}" for e in errs)
              + f"; RMSE {np.sqrt(np.mean(np.square(errs))):.2f} points; sampling at the same cost {se:.2f}")
        sys.stdout.flush()
