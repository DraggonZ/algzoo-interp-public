"""On full random inputs: the vote an input places when it arrives, and what happens to the old second's vote.

At each step t >= 2: x = new input, M = largest so far, m = second so far (before x). Record the centered score at slot 9
(the newest) right after the step, minus the best other slot. And when x becomes the new second, the change at the old
second's slot (after the step versus one slot further down before the step, i.e. what sliding alone would give).
"""
import numpy as np

from algzoo.core import example

w_hi, w_hh, w_oh = (np.asarray(p, np.float64) for p in example())
Wc = w_oh - w_oh.mean(0)
rng = np.random.default_rng(9)
N = 200000
X = rng.standard_normal((N, 10))
hh = np.zeros((N, 16)); prevS = None
recs = []
demote = []
for t in range(10):
    hh = np.maximum(0, hh @ w_hh.T + X[:, t, None] * w_hi)
    S = hh @ Wc.T
    if t >= 2:
        srt = np.sort(X[:, :t], 1)
        M, m = srt[:, -1], srt[:, -2]
        pos2 = np.argsort(X[:, :t], 1)[:, -2]
        x = X[:, t]
        others = np.delete(S, 9, 1).max(1)
        recs.append(np.c_[x, M, m, S[:, 9] - others])
        new2 = (x > m) & (x < M)
        # old second's slot: was at slot pos2 + (9 - (t-1)) before, now at pos2 + (9 - t)
        before = prevS[np.arange(N), pos2 + (9 - (t - 1))]
        after = S[np.arange(N), pos2 + (9 - t)]
        demote.append(np.c_[new2, after - before, x - m])
    prevS = S
R = np.vstack(recs)
x, M, m, v = R.T
print("vote at the newest slot (minus best other) by where the new input falls:")
for lab, k in [("new largest (x > M)", x > M), ("new second (m < x < M)", (x > m) & (x < M)), ("below second (x < m)", x < m)]:
    print(f"  {lab}: median {np.median(v[k]):6.1f}; positive {np.mean(v[k] > 0):.2f}")
print("for inputs below the largest: vote by x minus the current second (bins):")
k = x < M
for lo, hi in [(-3, -1), (-1, -0.3), (-0.3, -0.1), (-0.1, 0), (0, 0.1), (0.1, 0.3), (0.3, 1), (1, 3)]:
    b = k & (x - m >= lo) & (x - m < hi)
    print(f"  x - second in [{lo:+.1f},{hi:+.1f}): median vote {np.median(v[b]):6.1f}, newest is top {np.mean(v[b] > 0):.2f}, n {b.sum()}")
D = np.vstack(demote)
nd = D[:, 0] > 0
print(f"\nold second's score change when a new second arrives: median {np.median(D[nd, 1]):.1f}; when an input below the second arrives: {np.median(D[~nd, 1]):.1f}")
