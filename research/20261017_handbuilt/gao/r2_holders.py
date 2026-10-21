"""How do the networks that erase two places after a record do it? For new102 and new106 (and ARC at r+1 for
comparison): lists as in unerase_across.py; each neuron's share of the r+k number's effect (real run minus run with -0.5
at r+k), median absolute value just before the overtaking (T-1) and at the overtaking step (T), and that neuron's output
weight at position r+k relative to its others."""
from pathlib import Path
import numpy as np
from algzoo.core import load, example
import simlib
from simlib import lists
def net(k):
    return example() if k == "ARC" else load(str(next(Path("/data/output/research/algzoo/20261016_run3/models").glob(f"2nd_argmax_16_10_*_{k}.pth"))))
X = lists(41, 400_000); B = len(X); o = np.argsort(-X, 1); lar, sec, thi = o[:, 0], o[:, 1], o[:, 2]
m1p = np.stack([np.max(np.c_[np.zeros((B, 2)), X[:, :t]], 1) for t in range(10)], 1); rec = X > m1p
T = np.full(B, -1)
for t in range(9, -1, -1):
    T = np.where(rec[:, t] & (t > sec), t, T)
np.set_printoptions(precision=1, suppress=True, linewidth=180)
for name, k in (("ARC", 1), ("102", 2), ("106", 2)):
    P = tuple(np.asarray(p, float) for p in net(name)); P = (P[0].reshape(-1), P[1], P[2])
    sel = rec[np.arange(B), sec] & (thi == sec + k) & (T > sec + k + 1) & (lar == T) & (sec + k <= 7)
    Xs, Ts, ys = X[sel], T[sel], sec[sel]; Xt = Xs.copy(); Xt[np.arange(len(Xs)), ys + k] = -0.5
    H = simlib.run(Xs, params=P)[2]; Ht = simlib.run(Xt, params=P)[2]; D = H - Ht; i = np.arange(len(Xs))
    before, at = np.median(np.abs(D[i, Ts - 1]), 0), np.median(np.abs(D[i, Ts]), 0)
    print(f"{name}, number at record + {k} ({len(Xs)} lists): per-neuron share before / at the overtaking")
    print("   before:", before); print("   at:    ", at)
    print("   neurons whose share drops by over 80%:", [n for n in range(16) if before[n] > 0.3 and at[n] < 0.2 * before[n]])
