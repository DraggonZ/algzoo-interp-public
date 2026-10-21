"""Where does the trained network's margin come from when a record is overtaken by a slightly larger number?
Lists: a record at r (the second-largest at the end, above 0.3), overtaken at q > r by a number less than 5% larger,
which stays the largest. Margin M = score[r] - score[q] (the right answer over the new largest's position).
Report: median M; and its change per unit when the old record x_r alone is nudged, and when the new largest x_q alone
is nudged, split by neuron group (final state times readout). Trained network; v18 at gains 10,000 and at gains 10."""
import numpy as np
from algzoo.core import example
from algzoo.handbuilt_v18 import build_v18
rng = np.random.default_rng(17); X = rng.standard_normal((3000000, 10)); N = len(X)
def final(P, X):
    wi, wh, wo = (np.asarray(p, float) for p in P); h = np.zeros((len(X), len(wi)))
    for t in range(10): h = np.maximum(h @ wh.T + X[:, t:t + 1] * wi, 0)
    return h
I = build_v18().idx
g18 = {"memory": [I[f"mem{j}"] for j in range(8)], "hiding": [I[k] for k in ("clock0", "clock1", "clock2", "flip")],
       "store, detector, gates": [I[k] for k in ("store", "detect", "keep", "flag")]}
nets = {"trained": (example(), {"memory": [0, 3, 10, 11, 12, 13, 14, 15], "hiding": [1, 6, 7, 8], "store, detector, gates": [2, 4, 5, 9]}),
        "v18 gains 10,000": (build_v18().params, g18),
        "v18 gains 10": (build_v18(g=10, a=10, w=10, lam=10.0, c=-9.0, eps=0.1).params, g18)}
pairs = [(2, 5), (3, 6), (4, 7)]
eps = 1e-3
for name, (P, G) in nets.items():
    wo = np.asarray(P[2], float); print(f"== {name}")
    for r, q in pairs:
        pre = np.maximum(X[:, :r].max(1), 0)
        ok = (X[:, r] > pre) & (X[:, r] > 0.3) & (X[:, r + 1:q].max(1) < X[:, r]) & (X[:, q] > X[:, r]) & (X[:, q] < 1.05 * X[:, r])
        if q < 9: ok &= X[:, q + 1:].max(1) < X[:, r]
        Xs = X[ok]; H0 = final(P, Xs); L0 = H0 @ wo.T; M0 = L0[:, r] - L0[:, q]
        out = [f"r{r} q{q} (n={ok.sum()}): median margin {np.median(M0):7.2f}"]
        for lab, pos in (("nudge old record", r), ("nudge new largest", q)):
            Xp = Xs.copy(); Xp[:, pos] += eps; dH = (final(P, Xp) - H0) / eps
            parts = {k: np.median(dH[:, gi] @ (wo[r, gi] - wo[q, gi])) for k, gi in G.items()}
            tot = np.median(dH @ (wo[r] - wo[q]))
            out.append(f"{lab}: total {tot:7.1f} (" + ", ".join(f"{k} {v:7.1f}" for k, v in parts.items()) + ")")
        print("   " + " | ".join(out))
