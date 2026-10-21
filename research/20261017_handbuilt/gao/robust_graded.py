"""Pilot: does 'a right idea slightly off scores below a wrong idea' hold on the least
fragile twin (trained scale: g = a = w = 10, lam 10, vote gain 1 - c = 10, tie-breaker eps 0.1 = 1% of a vote;
robust_twin.log: 92.0%, survives 0.01% noise)? Same spoils as twin_graded.py, with the gate spoil as the same 6% change
of the vote gain (1 - c) and the store gain 2% off. All 16 replaced. Also the first-order prediction."""
import numpy as np
src = open("twin_graded.py").read(); src = src[:src.index("import os\ngrid")]
src = src.replace("g_, w_, a_ = (float(v) for v in sys.argv[1:4]) if len(sys.argv) > 3 else (1e4, 1e4, 1000.0)", "g_, w_, a_ = 10.0, 10.0, 10.0")
src = src.replace("build_v18(g=g_, w=w_, a=a_)", "build_v18(g=g_, w=w_, a=a_, lam=10.0, c=-9.0, eps=0.1)")
src = src.replace("s_, c_, eps, beta_ = 20.0, 0.75, 1e-4, 0.5", "s_, c_, eps, beta_ = 20.0, -9.0, 0.1, 0.5")
exec(src)
r = np.arange(len(X)); _, Pre, _ = run(X, params=P); WH, WO = P[1], P[2]
k = L.argmax(1); j = np.argsort(L, 1)[:, -2]; lead = L[r, k] - L[r, j]; y = X.argsort(1)[:, -2]
g = WO[k] - WO[j]; Gt = np.zeros_like(H)
for t in range(9, -1, -1):
    Gt[:, t] = g; g = (g * (Pre[:, t] > 0)) @ WH
cases = [("exact rules", {}), ("misreading: clock height from the largest", {"height": "largest"}), ("misreading: flip term left out", {"flip": False}),
         ("store gain 2% off", {"sg": 20.4}), ("clock angle 45.5 deg", {"deg": 45.5}), ("vote gain 6% low (as the 0.765 cutoff)", {"c": 1 - 10 * 0.94})]
for label, kw in cases:
    E = rules(**kw); M = memory(E); sims = {n: (M[:, :, MEM.index(n)] if n in MEM else E[n]) for n in range(16)}
    err = np.stack([sims[n] for n in range(16)], -1) - H; d = (Gt * err).sum((1, 2))
    meas = (run(X, sims, params=P)[0].argmax(1) == y).mean(); pred = np.where(lead + d < 0, j == y, k == y).mean()
    agree = (run(X, sims, params=P)[0].argmax(1) == k).mean()
    print(f"{label:44s}: measured {meas:.3f}; agreement with the twin {agree:.3f}; first-order predicted {pred:.3f}", flush=True)
