"""Question (follows tail_cancel.py): does the trained network keep an overtaken record's wave at positions after the
overtaking? Nudge the record at r and the overtaking number at q by the same eps: the new record's jump is unchanged,
the old record's jump grows by eps (and the largest so far by eps from r on). Final-score change per position, by
group (memory 0, 3, 10-15; cancelling 1, 6, 7, 8; other 2, 4, 5, 9), median over lists. Same for v18 as a reference:
v18 writes the old wave only at positions before the overtaking."""
import numpy as np
from algzoo.core import example
from algzoo.handbuilt_v18 import build_v18
rng = np.random.default_rng(3); X = rng.standard_normal((1500000, 10))
def final(P, X):
    wi, wh, wo = P; h = np.zeros((len(X), len(wi)))
    for t in range(10): h = np.maximum(h @ wh.T + X[:, t:t + 1] * wi, 0)
    return h
T = [np.asarray(p, float) for p in example()]
net = build_v18(); I = net.idx; V = [np.asarray(p, float) for p in net.params]
groups = {"trained": (T, {"memory": [0, 3, 10, 11, 12, 13, 14, 15], "cancel": [1, 6, 7, 8], "other": [2, 4, 5, 9]}),
          "v18": (V, {"memory": [I[f"mem{j}"] for j in range(8)], "clock+flip": [I[k] for k in ("clock0", "clock1", "clock2", "flip")],
                      "other": [I[k] for k in ("store", "detect", "keep", "flag")]})}
eps = 1e-3
for r, q in [(1, 4), (2, 5), (2, 7), (4, 7)]:
    pre = np.maximum(X[:, :r].max(1), 0)
    ok = (X[:, r] > pre + 0.05) & (X[:, r] > 0.3) & (X[:, r + 1:q].max(1) < X[:, r]) & (X[:, q] > X[:, r] + 0.05)
    if q < 9: ok &= X[:, q + 1:].max(1) < X[:, q]
    Xs = X[ok]; Xp = Xs.copy(); Xp[:, r] += eps; Xp[:, q] += eps
    print(f"record at {r}, overtaken at {q} (n={ok.sum()}); positions 0-9")
    for name, (P, G) in groups.items():
        dh = (final(P, Xp) - final(P, Xs)) / eps; wo = P[2]
        for nm, g in G.items():
            print(f"   {name:7s} {nm:10s}", " ".join(f"{x:7.1f}" for x in np.median(dh[:, g] @ wo[:, g].T, 0)))
        tot = np.median(dh @ wo.T, 0)
        print(f"   {name:7s} {'total':10s}", " ".join(f"{x:7.1f}" for x in tot))
