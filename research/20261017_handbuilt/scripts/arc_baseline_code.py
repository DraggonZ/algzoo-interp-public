"""Hypothesis: each position's final score is (largest before it) + (its excess over that), i.e. about its own value
for ordinary numbers. Test on positions p = 1..7 that are not new largests and not next to one, with no new largest
after p (so nothing later rewrites the code): regress the score (minus the mean score) on x_p, the largest before p
(m1), and the second-largest before p (m2), separately for positive and negative x_p."""
import numpy as np
from algzoo.core import example
wi, wh, wo = (np.asarray(w, dtype=float) for w in example()); wi = wi.reshape(-1)
X = np.random.default_rng(44).standard_normal((300_000, 10)); B = len(X)
h = np.zeros((B, 16))
for t in range(10): h = np.maximum(h @ wh.T + X[:, t:t + 1] * wi, 0)
s = h @ wo.T; s -= s.mean(1, keepdims=True)
srt = [np.sort(np.c_[np.zeros((B, 2)), X[:, :t]], 1) for t in range(10)]
m1 = np.stack([q[:, -1] for q in srt], 1); m2 = np.stack([q[:, -2] for q in srt], 1); rec = X > m1
for sign, name in ((1, "positive, below the largest"), (-1, "negative")):
    print(f"--- x_p {name}")
    for p in range(1, 8):
        m = ~rec[:, p] & ~rec[:, p - 1] & ~rec[:, p + 1] & ~rec[:, p + 1:].any(1) & (sign * X[:, p] > 0) & (m1[:, p] > 0)
        A = np.c_[X[m, p], m1[m, p], m2[m, p], X[m, 9]]
        co = np.linalg.lstsq(np.c_[A, np.ones(m.sum())], s[m, p], rcond=None)[0]
        r2 = 1 - ((s[m, p] - np.c_[A, np.ones(m.sum())] @ co) ** 2).mean() / s[m, p].var()
        print(f"  p={p} ({m.sum():6d}): x_p {co[0]:6.1f}  m1 {co[1]:6.1f}  m2 {co[2]:6.1f}  x_9 {co[3]:6.1f}  R2 {r2:.2f}")

print("--- new-largest positions. old largest r (overtaken later, exactly two new largests after r's predecessor...):")
recpos = [np.flatnonzero(r) for r in rec]
nrec = rec.sum(1)
# lists with >= 2 new largests; r = second-to-last, L = last; P_prev = value of the one before r (0 if none)
ok = nrec >= 2
r = np.array([q[-2] if len(q) >= 2 else -1 for q in recpos]); L = np.array([q[-1] if len(q) >= 1 else -1 for q in recpos])
pp = np.array([X[i, q[-3]] if len(q) >= 3 else 0.0 for i, q in enumerate(recpos)])
i = np.arange(B)
for name, pos in (("score at old largest r", r), ("score at current largest L", L)):
    print(f"  {name}: coefficients on old largest P, current largest x_L, previous record P_prev, second-largest-before-L")
    for (lo, hi) in ((1, 3), (3, 5), (5, 8)):
        m = ok & (L - r >= 2) & (pos >= lo) & (pos < hi) & (L <= 7)
        m2L = np.array([np.sort(np.r_[0, 0, X[j, :L[j]]])[-2] for j in np.flatnonzero(m)])
        A = np.c_[X[i, r][m], X[i, L][m], pp[m], m2L, np.ones(m.sum())]
        co = np.linalg.lstsq(A, s[i, pos][m], rcond=None)[0]
        r2 = 1 - ((s[i, pos][m] - A @ co) ** 2).mean() / s[i, pos][m].var()
        print(f"    position {lo}-{hi - 1} ({m.sum():6d}): P {co[0]:6.1f}  x_L {co[1]:6.1f}  P_prev {co[2]:6.1f}  m2 {co[3]:6.1f}  R2 {r2:.2f}")
