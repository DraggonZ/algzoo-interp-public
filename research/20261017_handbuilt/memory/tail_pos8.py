"""Question: overtakings at 6 or 7 shortly after the record put up to +28 per unit on position 8 (tail_routes_all.log).
Which routes carry it? Per placement (q = 6, 7), the route split (as tail_routes_all.py) at position 8, nudge x_r alone
and x_r with x_q; the other neurons' readout split by neuron. 1,500,000 lists, seed 3."""
import numpy as np
from decomp import decompose, SOURCES
from memlib import OTH, RO
from tail_routes_all import split, routes, X, pre_all
for nudge in ("r", "both"):
    print(f"=== nudge {nudge}: position 8")
    print("   r q  " + " ".join(f"{k[:14]:>14s}" for k in routes) + " | readout by neuron " + " ".join(f"{n:>5d}" for n in OTH))
    for q in (6, 7):
        for r in range(0, q):
            ok = (X[:, r] > pre_all[:, r] + 0.05) & (X[:, r] > 0.3) & (X[:, q] > X[:, r] + 0.05)
            if q > r + 1: ok &= X[:, r + 1:q].max(1) < X[:, r]
            ok &= X[:, q + 1:].max(1) < X[:, q]
            Xs = X[ok][:20_000]; dX = np.zeros_like(Xs); dX[:, r] = 1
            if nudge == "both": dX[:, q] = 1
            R = decompose(Xs, dX); out = split(R, q); m = R["same"]
            per = (R["dH"][m][:, 9][:, OTH] * RO[8][None]).mean(0)
            print(f"   {r} {q}  " + " ".join(f"{out[k][8]:14.1f}" for k in routes) + " | " + " ".join(f"{x:5.1f}" for x in per))
