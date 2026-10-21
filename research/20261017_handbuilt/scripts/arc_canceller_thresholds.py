"""Do the trained network's four cancelling neurons switch off at the same input? On random lists, for steps 3-9 where
a canceller is on, fit its value as  g * (alpha * m1_prev - x) + rest, where m1_prev is the largest before this step;
the canceller switches off near x = alpha * m1_prev (plus its other drive). Also: the input at which each turns off,
binned by the ratio x / m1_prev."""
import numpy as np
from algzoo.core import example
wi, wh, wo = (np.asarray(w, dtype=float) for w in example()); wi = wi.reshape(-1)
X = np.random.default_rng(32).standard_normal((100_000, 10)); B, n = X.shape
H = []; h = np.zeros((B, 16))
for t in range(n): h = np.maximum(h @ wh.T + X[:, t:t + 1] * wi, 0); H.append(h)
H = np.stack(H, 1)
m1p = np.stack([np.max(np.c_[np.zeros((B, 1)), X[:, :t]], 1) for t in range(n)], 1)
sel = (slice(None), slice(3, 10))
x, m = X[sel].ravel(), m1p[sel].ravel()
print("share of steps on, by input as a fraction of the largest so far (rows: canceller 1, 6, 7, 8)")
bins = np.array([-1, 0.5, 0.7, 0.8, 0.9, 0.95, 1.0, 1.05, 1.1, 1.3, 9]); r = x / np.maximum(m, 1e-9)
print("ratio bins      ", " ".join(f"{lo:>5.2f}" for lo in bins[:-1]))
for k in (1, 6, 7, 8):
    v = H[:, 3:10, k].ravel()
    on = [(v[(r >= lo) & (r < hi) & (m > 0.3)] > 0.05).mean() for lo, hi in zip(bins[:-1], bins[1:])]
    A = np.c_[x, m, np.ones_like(x)][v > 1]
    coef = np.linalg.lstsq(A, v[v > 1], rcond=None)[0]
    print(f"canceller {k}: on", " ".join(f"{o:5.2f}" for o in on), f"  fit: {coef[0]:.1f} x + {coef[1]:.1f} m1 -> alpha {-coef[1] / coef[0]:.2f}")
