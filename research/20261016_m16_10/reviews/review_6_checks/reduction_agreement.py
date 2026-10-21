"""Review 6, check A: does "only the three largest inputs matter" hold input by input, or only on average?

For random inputs, compare the network's answer on the full input with its answer on the reduced input (top three kept,
the other seven set to E[X | X < third] = -phi(m3)/Phi(m3), and the constant -0.46 used by the estimate). Report:
accuracy of each, how often the two answers agree, and the mistakes that the small inputs cause (full wrong, reduced right)
and fix (full right, reduced wrong). Also an independent sampled value of the constant-background reduction, to compare
with the quadrature values in reduce3_quad.log. CPU, fresh random inputs (seed 6), six trained models only.
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "2")
import numpy as np
import torch as th
from scipy.stats import norm

from algzoo.core import example, zoo

th.set_num_threads(2)
N = 2 ** 22
QUAD = {"example": 0.9530, 0: 0.9108, 1: 0.8947, 2: 0.9488, 3: 0.9317, 4: 0.8893}


def answers(params, X):
    w_hi, w_hh, w_oh = (th.as_tensor(np.asarray(q), dtype=th.float32) for q in params)
    out = []
    for xb in th.split(X, 2 ** 19):
        h = th.zeros(len(xb), len(w_hi))
        for t in range(10):
            h = th.relu(h @ w_hh.T + xb[:, t, None] * w_hi)
        out.append((h @ w_oh.T).argmax(-1))
    return th.cat(out)


g = np.random.default_rng(6)
X = g.standard_normal((N, 10)).astype(np.float32)
order = np.argsort(X, 1)
tgt = th.as_tensor(order[:, -2])
m3 = X[np.arange(N), order[:, -3]]
cm = (-norm.pdf(m3) / norm.cdf(m3)).astype(np.float32)
small = np.zeros_like(X, dtype=bool)
for j in range(7):
    small[np.arange(N), order[:, j]] = True
Xc = np.where(small, cm[:, None], X)
Xk = np.where(small, np.float32(-0.46), X)
X, Xc, Xk = th.as_tensor(X), th.as_tensor(Xc), th.as_tensor(Xk)
for name in ["example", 0, 1, 2, 3, 4]:
    p = example() if name == "example" else zoo(16, 10, name)
    a, c, k = answers(p, X), answers(p, Xc), answers(p, Xk)
    ra, rc, rk = (a == tgt), (c == tgt), (k == tgt)
    se = 100 * np.sqrt(rk.float().mean().item() * (1 - rk.float().mean().item()) / N)
    print(f"{name}: full {100 * ra.float().mean():.2f}%, reduced (cond. mean) {100 * rc.float().mean():.2f}%, (const -0.46) {100 * rk.float().mean():.2f}% "
          f"[quadrature {100 * QUAD[name]:.2f}, sampled SE {se:.2f}]; same answer full vs cond.-mean {100 * (a == c).float().mean():.2f}%; "
          f"small inputs cause {100 * (~ra & rc).float().mean():.2f} points of mistakes and fix {100 * (ra & ~rc).float().mean():.2f}", flush=True)
