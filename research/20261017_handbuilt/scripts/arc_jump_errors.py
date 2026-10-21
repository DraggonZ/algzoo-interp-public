"""If the revealed old largest votes about x_r - 0.4 * P_prev, the trained network should err more when the second is
a revealed old largest whose previous record P_prev was close to it. Split those lists by P_prev / x_r."""
import numpy as np
from algzoo.core import example
wi, wh, wo = (np.asarray(w, dtype=float) for w in example()); wi = wi.reshape(-1)
X = np.random.default_rng(36).standard_normal((1_000_000, 10)); B = len(X)
h = np.zeros((B, 16))
for t in range(10): h = np.maximum(h @ wh.T + X[:, t:t + 1] * wi, 0)
P = (h @ wo.T).argmax(1); o = X.argsort(1); sec, thi = o[:, -2], o[:, -3]
m1p = np.stack([np.max(np.c_[np.zeros((B, 1)), X[:, :t]], 1) for t in range(10)], 1); rec = X > m1p
idx = np.arange(B); revealed = rec[idx, sec] & (sec < o[:, -1])          # second was a new largest when it came
pprev = m1p[idx, sec]                                                   # largest before the second (0 if none positive)
gap3 = (X[idx, sec] - X[idx, thi]) / np.abs(X[idx, sec])
print("second = revealed old largest, third-largest within 20% of it; error rate by P_prev / second")
for lo, hi in ((0, 1e-9), (1e-9, 0.5), (0.5, 0.8), (0.8, 0.9), (0.9, 1.0)):
    m = revealed & (pprev / X[idx, sec] >= lo) & (pprev / X[idx, sec] < hi) & (gap3 < 0.2) & (X[idx, thi] > 0)
    print(f"  P_prev/second in [{lo:.1f},{hi:.1f}): lists {m.sum():6d}  wrong {(P[m] != sec[m]).mean():.3f}  "
          f"third named {(P[m] == thi[m]).mean():.3f}  third is P_prev {(rec[idx, thi] & (thi < sec))[m].mean():.2f}")
print("same lists, third within 10% of the second: is the third the previous record or another number?")
for lo, hi in ((0, 0.03), (0.03, 0.1)):
    for name, k in (("third = previous record", rec[idx, thi] & (thi < sec)), ("third = other number", ~(rec[idx, thi] & (thi < sec)))):
        m = revealed & (gap3 >= lo) & (gap3 < hi) & (X[idx, thi] > 0) & k
        print(f"  gap [{lo:.2f},{hi:.2f}) {name:24s} lists {m.sum():6d}  wrong {(P[m] != sec[m]).mean():.3f}")
