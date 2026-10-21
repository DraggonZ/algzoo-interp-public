"""Does the first-order (attribution-patching) prediction explain the twin's simulation scores (Section 4: a gain 2% off
scores below conceptual misreadings)? Each spoiled account from twin_graded.py, all 16 neurons replaced; predicted
lead change sum_{t,n} g_{t,n} (sim - true) with g from a backward pass through the twin's own run. Also the error's
plain size (root-mean-square relative to typical values) for contrast. Args: clock, flag, detector gains."""
import sys, os
import numpy as np
src = open("twin_graded.py").read(); src = src[:src.index("import os\ngrid")]
exec(src)
r = np.arange(len(X)); _, Pre, _ = run(X, params=P); WH, WO = P[1], P[2]
k = L.argmax(1); j = np.argsort(L, 1)[:, -2]; lead = L[r, k] - L[r, j]; y = X.argsort(1)[:, -2]
g = WO[k] - WO[j]; Gt = np.zeros_like(H)
for t in range(9, -1, -1):
    Gt[:, t] = g; g = (g * (Pre[:, t] > 0)) @ WH
cases = [("exact rules", {}), ("misreading: clock height from the largest", {"height": "largest"}), ("misreading: flip term left out", {"flip": False}),
         ("store gain 2% off", {"sg": 20.4}), ("clock angle 45.5 deg", {"deg": 45.5}), ("gate cutoff 0.765", {"c": 0.765})]
for label, kw in cases:
    E = rules(**kw); M = memory(E); sims = {n: (M[:, :, MEM.index(n)] if n in MEM else E[n]) for n in range(16)}
    Sim = np.stack([sims[n] for n in range(16)], -1); err = Sim - H
    d = (Gt * err).sum((1, 2)); pred = np.where(lead + d < 0, j == y, k == y).mean()
    meas = (run(X, sims, params=P)[0].argmax(1) == y).mean()
    size = np.sqrt(np.mean((err / typ) ** 2))
    print(f"{label:44s}: measured {meas:.3f}; first-order predicted {pred:.3f}; error size (rms / typical) {size:.2e}; median |predicted lead change| / lead {np.median(np.abs(d) / np.maximum(lead, 1e-12)):.2f}", flush=True)
