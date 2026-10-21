"""w13b a0: who reads the clocks in w12b, what the clocks read, and the late gate's value at ages after a record."""
import numpy as np
from algzoo.handbuilt_w12b import build_w12b, CLOCKS

wi, wh, wo, idx, info = build_w12b()
name = {v: k for k, v in idx.items()}
C = [idx[k] for k in CLOCKS]
print("neurons:", [name[i] for i in range(len(wi))])
for c in C:
    print(f"clock row {name[c]:7s} wi {wi[c]:+.3g}", {name[k]: f"{wh[c, k]:+.4g}" for k in np.flatnonzero(np.abs(wh[c]) > 1e-12)})
print("non-clock readers of clocks:")
for r in range(len(wi)):
    if r in C: continue
    w = wh[r, C]
    if np.abs(w).max() > 1e-12: print(f"  {name[r]:7s}", np.round(w, 6))
print("readout of clocks (rows = positions):"); print(np.round(wo[:, C], 5))
print("readout weights (all neurons) at positions:")
for k in range(len(wi)):
    if np.abs(wo[:, k]).max() > 1e-12: print(f"  {name[k]:7s}", np.round(wo[:, k], 4))
LG = idx["late"]
print("late gate row:", {name[k]: f"{wh[LG, k]:+.4g}" for k in np.flatnonzero(np.abs(wh[LG]) > 1e-12)}, "wi", wi[LG])
# simulate a list with one record at step 3, then small numbers
x = np.array([[0.5, 0.2, -0.3, 2.0, -1.0, -0.5, 0.1, -1.2, 0.3, -0.4]])
h = np.zeros((1, len(wi))); rows = []
for t in range(10):
    h = np.maximum(h @ wh.T + x[:, t:t + 1] * wi, 0); rows.append(h[0].copy())
rows = np.array(rows)
for nm in ("store", "detect", "late", "c7", "c1", "c6", "h7", "keep", "Q"):
    print(f"{nm:7s}", np.round(rows[:, idx[nm]], 4))
for c in C:
    print(f"{name[c]:7s}-c7", np.array2string(rows[:, c] - rows[:, idx['c7']], precision=3))
