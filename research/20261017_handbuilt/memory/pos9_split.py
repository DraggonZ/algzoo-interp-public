"""Question: at position 9, which non-memory neurons' own readout outweighs the memory's level? For all placements of
tail_routes_all.py (joint nudge and record alone), the other neurons' readout at position 9 split by neuron, mean
per unit, averaged over placements. 1,500,000 lists, seed 3."""
import numpy as np
from decomp import decompose
from memlib import OTH, RO
from tail_routes_all import X, pre_all
for nudge in ("both", "r"):
    rows = []
    for q in range(1, 9):
        for r in range(0, q):
            ok = (X[:, r] > pre_all[:, r] + 0.05) & (X[:, r] > 0.3) & (X[:, q] > X[:, r] + 0.05) & (X[:, q + 1:].max(1) < X[:, q])
            if q > r + 1: ok &= X[:, r + 1:q].max(1) < X[:, r]
            Xs = X[ok][:20_000]; dX = np.zeros_like(Xs); dX[:, r] = 1
            if nudge == "both": dX[:, q] = 1
            R = decompose(Xs, dX); m = R["same"]
            rows.append((R["dH"][m][:, 9][:, OTH] * RO[9][None]).mean(0))
    v = np.mean(rows, 0)
    print(f"nudge {nudge}: readout at position 9 by neuron (mean over {len(rows)} placements): " + " ".join(f"{n}:{x:+.1f}" for n, x in zip(OTH, v)) + f" | hiding 1,6,7,8 together {v[[0, 4, 5, 6]].sum():+.1f}, all {v.sum():+.1f}")
