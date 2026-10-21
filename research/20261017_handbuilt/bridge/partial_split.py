"""What makes up the 63% (positive non-record numbers one step after a record at which the hiding group is partly off)?
Split by the previous largest p before the record: none positive (p = 0), small (0 < p <= 0.2), larger; and, for
p > 0.2, by the arriving number x relative to p."""
import numpy as np
from algzoo.core import example
wi, wh, wo = (np.asarray(p, float) for p in example())
rng = np.random.default_rng(7); X = rng.standard_normal((400000, 10)); N = len(X)
h = np.zeros((N, 16)); HS = []
for t in range(10): h = np.maximum(h @ wh.T + X[:, t:t + 1] * wi, 0); HS.append(h)
HS = np.stack(HS, 1)
pm = np.maximum.accumulate(np.c_[np.zeros(N), np.maximum(X, 0)], 1)       # largest so far, 0 before any positive
rec = X > pm[:, :-1]
off = (HS[:, :, [1, 6, 7, 8]] == 0).sum(2); part = (off > 0) & (off < 4)
P = []; Q = []; PART = []
for r in range(0, 9):
    m = rec[:, r] & (X[:, r + 1] > 0) & ~rec[:, r + 1]
    P.append(pm[m, r]); Q.append(X[m, r + 1]); PART.append(part[m, r + 1])
p = np.concatenate(P); x = np.concatenate(Q); pt = np.concatenate(PART)
print(f"all: share partly off {pt.mean():.2f} (n={len(pt)})")
for lab, s in [("no positive number before the record (p = 0)", p == 0), ("0 < p <= 0.2", (p > 0) & (p <= 0.2)), ("p > 0.2", p > 0.2)]:
    print(f"   {lab:46s}: {s.mean():.2f} of cases, partly off {pt[s].mean():.2f}")
s = p > 0.2; q = x[s] / p[s]
for a, b in [(0, 0.5), (0.5, 0.85), (0.85, 1.0), (1.0, 9)]:
    k = (q >= a) & (q < b); print(f"      p > 0.2 and x / p in [{a}, {b}): {k.mean():.2f} of those cases, partly off {pt[s][k].mean():.2f}")
