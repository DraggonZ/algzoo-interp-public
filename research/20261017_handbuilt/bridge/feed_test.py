"""Question: is the broad hiding wave that the trained memory holds for the current largest fed into the memory by
the cancelling neurons step by step (like v18's forward copies from its clock), rather than written once by the
detector? Test: zero the cancelling neurons' weights into the memory (wh[memory, cancelling] = 0) and compare the
memory's part of a record's profile (single nudge of a record that stays largest; median over lists) with the
original, and with zeroing the detector's weights into the memory instead."""
import numpy as np
from algzoo.core import example
wi, wh0, wo = (np.asarray(p, float) for p in example())
MEM = [0, 3, 10, 11, 12, 13, 14, 15]; CAN = [1, 6, 7, 8]
rng = np.random.default_rng(2); X = rng.standard_normal((600000, 10))
def final(X, wh):
    h = np.zeros((len(X), 16))
    for t in range(10): h = np.maximum(h @ wh.T + X[:, t:t + 1] * wi, 0)
    return h
variants = {"as trained": wh0.copy()}
w = wh0.copy(); w[np.ix_(MEM, CAN)] = 0; variants["no cancelling -> memory"] = w
w = wh0.copy(); w[MEM, 4] = 0; variants["no detector -> memory"] = w
eps = 1e-3
for r in (2, 4):
    pre = np.maximum(X[:, :r].max(1), 0)
    ok = (X[:, r] > pre + 0.05) & (X[:, r] > 0.3) & (X[:, r + 1:].max(1) < X[:, r])
    Xs = X[ok]; Xp = Xs.copy(); Xp[:, r] += eps
    print(f"record at {r}, stays largest (n={ok.sum()}): memory's part of the final scores, positions 0-9")
    for nm, wh in variants.items():
        dh = (final(Xp, wh) - final(Xs, wh)) / eps
        print(f"   {nm:24s}", " ".join(f"{v:7.1f}" for v in np.median(dh[:, MEM] @ wo[:, MEM].T, 0)))
