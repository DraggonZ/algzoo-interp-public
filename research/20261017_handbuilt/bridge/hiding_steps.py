"""What do the four hiding neurons hold k = 1, 2, 3 steps after a record? Lists with a record at r (2..5, beats the
previous largest p > 0.2 by 0.05) and negative numbers between (so the four stay in their linear range before step
r + k); fit each neuron's pre-activation at r + k on (p, new largest m1, arriving number x). Analysis only."""
import numpy as np
from algzoo.core import example
wi, wh, wo = (np.asarray(p, float) for p in example())
rng = np.random.default_rng(16); X = rng.standard_normal((3000000, 10)); N = len(X)
h = np.zeros((N, 16)); PRE = []
for t in range(10): pre = h @ wh.T + X[:, t:t + 1] * wi; h = np.maximum(pre, 0); PRE.append(pre)
for k in (1, 2, 3):
    rows = []
    for r in range(2, 6):
        p = np.maximum(X[:, :r].max(1), 0); ok = (X[:, r] > p + 0.05) & (p > 0.2)
        if k > 1: ok &= (X[:, r + 1:r + k] < 0).all(1)
        rows.append(np.c_[p[ok], X[ok, r], X[ok, r + k], PRE[r + k][ok][:, [1, 6, 7, 8]]])
    R = np.concatenate(rows); A = R[:, :3]
    out = []
    for i, n in enumerate((1, 6, 7, 8)):
        c = np.linalg.lstsq(A, R[:, 3 + i], rcond=None)[0]; e = R[:, 3 + i] - A @ c
        out.append(f"{n}: {c[0]:6.2f} p {c[1]:+6.2f} m1 {c[2]:+6.2f} x ({1 - e.var() / R[:, 3 + i].var():.2f})")
    print(f"{k} step(s) after the record (n={len(R)}): " + " | ".join(out))
