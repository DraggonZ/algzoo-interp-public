"""Prediction from a jump-driven clock: a revealed old largest x_r votes about x_r - 0.7 * P_prev, where P_prev is the
largest before it. Test on the trained network with small nudges on random lists that have new largests at a < r < r'
(P_prev = x_a, old largest x_r, current largest at r', no other new largests after a). Compare the change in position
r's score (relative to the mean score) per unit nudge of x_a and of x_r. Control: nudging a non-record number."""
import numpy as np
from algzoo.core import example
wi, wh, wo = (np.asarray(w, dtype=float) for w in example()); wi = wi.reshape(-1)
def sc(X):
    h = np.zeros((len(X), 16))
    for t in range(10): h = np.maximum(h @ wh.T + X[:, t:t + 1] * wi, 0)
    s = h @ wo.T; return s - s.mean(1, keepdims=True)
rng = np.random.default_rng(35); X = rng.standard_normal((400_000, 10)); B = len(X)
m1p = np.stack([np.max(np.c_[np.zeros((B, 1)), X[:, :t]], 1) for t in range(10)], 1); rec = X > m1p
recpos = [np.flatnonzero(r) for r in rec]
rows = [(i, rp[-3], rp[-2], rp[-1]) for i, rp in enumerate(recpos) if len(rp) >= 3 and rp[-1] - rp[-2] >= 2 and rp[-2] - rp[-3] >= 2 and rp[-1] <= 8]
i, a, r, r2 = map(np.array, zip(*rows)); n = len(i); eps = 1e-3
Xs = X[i]; s0 = sc(Xs)[np.arange(n), r]
def nudge(pos):
    Y = Xs.copy(); Y[np.arange(n), pos] += eps; return (sc(Y)[np.arange(n), r] - s0) / eps
ga, gr = nudge(a), nudge(r)
# a non-record, non-negative-filler number between a and r as control
ctrl = np.array([next((p for p in range(aa + 1, rr) if 0 < X[ii, p]), -1) for ii, aa, rr in zip(i, a, r)])
m = ctrl >= 0; gc = nudge(np.where(m, ctrl, a))
print(f"{n} lists (records at a < r < r', gaps >= 2, r' <= 8)")
print(f"d score_r / d x_r       mean {gr.mean():6.1f}  median {np.median(gr):6.1f}")
print(f"d score_r / d P_prev    mean {ga.mean():6.1f}  median {np.median(ga):6.1f}   ratio of means {ga.mean() / gr.mean():.2f}")
print(f"d score_r / d kept number between them (control, {m.sum()} lists)  mean {gc[m].mean():6.1f}")
for lo, hi in ((2, 3), (3, 5), (5, 9)):
    k = (r - a >= lo) & (r - a < hi)
    print(f"  gap r - a in [{lo},{hi}): ratio {ga[k].mean() / gr[k].mean():.2f}  ({k.sum()} lists)")
