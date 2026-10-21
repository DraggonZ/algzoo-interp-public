"""Question: one step after a new largest, do hiding neurons 6 and 8 compare the arriving number with the previous
largest (now the second-largest)? Lists with a record at r = 2..6 (beats the previous largest p > 0.2 by 0.05); the
number x at r + 1. Share of lists with neuron 6 / 8 / 1 / 7 off at step r + 1, binned by x / p. Also a least-squares fit
of each neuron's pre-activation at r + 1 on (p, new largest, x) (analysis only)."""
import numpy as np
from algzoo.core import example
wi, wh, wo = (np.asarray(p, float) for p in example())
rng = np.random.default_rng(13); X = rng.standard_normal((2000000, 10)); N = len(X)
H = np.zeros((N, 16)); PRE = []; HS = []
for t in range(10):
    pre = H @ wh.T + X[:, t:t + 1] * wi; H = np.maximum(pre, 0); PRE.append(pre); HS.append(H)
PRE = np.stack(PRE, 1); HS = np.stack(HS, 1)
rows = []
for r in range(2, 7):
    p = np.maximum(X[:, :r].max(1), 0); ok = (X[:, r] > p + 0.05) & (p > 0.2)
    rows.append((p[ok], X[ok, r], X[ok, r + 1], PRE[ok, r + 1], HS[ok, r + 1]))
p = np.concatenate([a[0] for a in rows]); m1 = np.concatenate([a[1] for a in rows]); x = np.concatenate([a[2] for a in rows])
pre = np.concatenate([a[3] for a in rows]); hs = np.concatenate([a[4] for a in rows])
bins = [-9, 0, 0.25, 0.5, 0.65, 0.75, 0.85, 0.95, 1.05, 9]; q = x / p
print("x / previous largest:  " + "  ".join(f"[{a:.2f},{b:.2f})" for a, b in zip(bins[:-1], bins[1:])))
for n in (6, 8, 1, 7):
    print(f"   neuron {n} off share: " + "  ".join(f"{(hs[(q >= a) & (q < b), n] == 0).mean():11.2f}" for a, b in zip(bins[:-1], bins[1:])))
A = np.c_[p, m1, x]
for n in (6, 8, 1, 7):
    c = np.linalg.lstsq(A, pre[:, n], rcond=None)[0]; e = pre[:, n] - A @ c
    print(f"   neuron {n} pre-activation at r+1 = {c[0]:6.2f} p + {c[1]:6.2f} new largest + {c[2]:6.2f} x  (variance matched {1 - e.var() / pre[:, n].var():.2f})")
