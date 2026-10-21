"""Question: how sharp is one kept number's vote in the trained network's scores, against v18's?
Nudge a kept number (not a record, at least 0.8 x the second-largest before it, so the gate keeps it) at position r
by +eps and measure the change of every position's final score per unit; median over random lists."""
import numpy as np
from algzoo.core import example
from algzoo.handbuilt_v18 import build_v18
rng = np.random.default_rng(1); X = rng.standard_normal((200000, 10))
def scores(P, X):
    wi, wh, wo = (np.asarray(p, float) for p in P); h = np.zeros((len(X), len(wi)))
    for t in range(X.shape[1]): h = np.maximum(h @ wh.T + X[:, t:t + 1] * wi, 0)
    return h @ wo.T
nets = {"trained": example(), "v18": build_v18().params}
eps = 1e-3
for r in (2, 4, 6):
    pre = X[:, :r]; m1 = np.maximum(pre.max(1), 0); m2 = np.sort(np.c_[pre, np.zeros(len(X))], 1)[:, -2]
    ok = (X[:, r] < m1) & (X[:, r] > 0.8 * np.maximum(m2, 0)) & (X[:, r] > 0.2) & (X[:, r + 1:].max(1) < m1)   # kept, largest stays
    Xs = X[ok]; Xp = Xs.copy(); Xp[:, r] += eps
    for name, P in nets.items():
        d = (scores(P, Xp) - scores(P, Xs)) / eps
        print(f"r={r} {name:8s} n={ok.sum():6d} median d score/dx:", " ".join(f"{v:6.2f}" for v in np.median(d, 0)))
