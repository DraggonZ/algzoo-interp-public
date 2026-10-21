"""Question: is part of an overtaken record's revealed vote written after the overtaking, and does it depend on the gap
between the record (r) and the overtaking (q)? Exact split (decomp.py, nudge x_r and x_q together, mean per unit) of
the score at r: the wave written up to step q, and the writes at q+2 by hiding 6 and 8 (one step after a record they
hold about 11-13 x the previous largest minus the number). Mean over placements with the same gap; lists as
tail_routes_all.py (1,500,000 lists, seed 3, up to 20,000 per placement, q <= 7 so that q+2 exists)."""
import numpy as np
from tail_routes_all import split, X, pre_all
from decomp import decompose
by_gap = {}
for q in range(1, 8):
    for r in range(0, q):
        ok = (X[:, r] > pre_all[:, r] + 0.05) & (X[:, r] > 0.3) & (X[:, q] > X[:, r] + 0.05) & (X[:, q + 1:].max(1) < X[:, q])
        if q > r + 1: ok &= X[:, r + 1:q].max(1) < X[:, r]
        Xs = X[ok][:20_000]; dX = np.zeros_like(Xs); dX[:, r] = 1; dX[:, q] = 1
        out = split(decompose(Xs, dX), q)
        by_gap.setdefault(q - r, []).append((out["total"][r], out["wave written while hidden"][r], out["hiding 6+8 at q+2"][r]))
print("gap q-r | placements | vote at r (total) | wave written while hidden | hiding 6+8 at q+2 | share of the vote from 6+8 at q+2")
for g, v in sorted(by_gap.items()):
    v = np.array(v); print(f"   {g}    |    {len(v)}      | {v[:, 0].mean():6.1f} | {v[:, 1].mean():6.1f} | {v[:, 2].mean():6.1f} | {v[:, 2].mean() / v[:, 0].mean():5.2f}")
