"""Which neurons carry the erasure in new106 and new102 (r+2 erasers), and in ARC (r+1)? Own-score response of a kept
number y at q = r + k (q = 4, 5) after a record at r, split per neuron: wo[q, n] x d h_final[n] / d y, median over lists
where the record stays the largest and over lists where it is later overtaken (as in erasure_across.py)."""
from pathlib import Path
import numpy as np
from algzoo.core import load, example
import simlib
def net(k):
    return example() if k == "ARC" else load(str(next(Path("/data/output/research/algzoo/20261016_run3/models").glob(f"2nd_argmax_16_10_*_{k}.pth"))))
X = np.random.default_rng(33).standard_normal((400_000, 10)); B = len(X); eps = 0.01
m1p = np.stack([np.max(np.c_[np.zeros((B, 2)), X[:, :t]], 1) for t in range(10)], 1); rec = X > m1p
np.set_printoptions(precision=1, suppress=True, linewidth=180)
for name, k in (("ARC", 1), ("106", 2), ("102", 2)):
    P = tuple(np.asarray(p, float) for p in net(name)); P = (P[0].reshape(-1), P[1], P[2]); wo = P[2]
    for q in (5,):
        r = q - k
        base = rec[:, r] & ~rec[:, r + 1:q + 1].any(1) & (X[:, q] > 0.5 * X[:, r]) & (X[:, q] < X[:, r] - 2 * eps)
        out = {}
        for lab, mm in (("stays", base & ~rec[:, q + 1:].any(1)), ("overtaken", base & rec[:, q + 1:].any(1))):
            Z = X[mm]; Z2 = Z.copy(); Z2[:, q] += eps
            dH = (simlib.run(Z2, params=P)[2][:, 9] - simlib.run(Z, params=P)[2][:, 9]) / eps
            out[lab] = np.median(dH * wo[q][None, :], 0)
        print(f"{name}, y at record + {k} (q={q}): per-neuron contribution to y's own score")
        print("   stays:     ", out["stays"]); print("   overtaken: ", out["overtaken"])
        d = out["overtaken"] - out["stays"]; print("   largest drops:", [(n, round(d[n], 1)) for n in np.argsort(d)[:4]])
