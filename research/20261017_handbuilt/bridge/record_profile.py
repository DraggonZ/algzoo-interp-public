"""Question: what does the trained memory hold for the current largest (a record that is never overtaken), position by
position, and what do the cancelling neurons subtract? Nudge the record by +eps; final-state change times readout,
summed over the memory neurons (0, 3, 10-15) and over the cancelling neurons (1, 6, 7, 8); median over lists.
Then the same for a record that is overtaken later (its vote revealed)."""
import numpy as np
from algzoo.core import example
wi, wh, wo = example()
rng = np.random.default_rng(2); X = rng.standard_normal((400000, 10))
def final(X):
    h = np.zeros((len(X), 16))
    for t in range(10): h = np.maximum(h @ wh.T + X[:, t:t + 1] * wi, 0)
    return h
MEM = [0, 3, 10, 11, 12, 13, 14, 15]; CAN = [1, 6, 7, 8]; REST = [2, 4, 5, 9]
eps = 1e-3
for r in (2, 4, 6):
    pre = np.maximum(X[:, :r].max(1), 0)
    for kind in ("stays largest", "overtaken later"):
        later = X[:, r + 1:].max(1)
        ok = (X[:, r] > pre + 0.05) & (X[:, r] > 0.3)
        ok &= (later < X[:, r]) if kind == "stays largest" else (later > X[:, r] + 0.05)
        Xs = X[ok]; Xp = Xs.copy(); Xp[:, r] += eps
        dh = (final(Xp) - final(Xs)) / eps
        print(f"record at {r}, {kind} (n={ok.sum()})")
        for nm, g in (("memory", MEM), ("cancel", CAN), ("other", REST)):
            prof = np.median(dh[:, g] @ wo[:, g].T, 0)
            print(f"   {nm:7s}", " ".join(f"{v:7.1f}" for v in prof))
        print("   total  ", " ".join(f"{v:7.1f}" for v in np.median(dh @ wo.T, 0)))
