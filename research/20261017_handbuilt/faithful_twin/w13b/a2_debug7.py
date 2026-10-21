"""w13b a2: why a second-largest at position 7 is never named (a1): scores and the readout's terms at position 7."""
import numpy as np
from algzoo.handbuilt_w13b import build_w13b

wi, wh, wo, idx, info = build_w13b()
name = {v: k for k, v in idx.items()}
xx = np.array([[0.1, -0.5, 0.3, -0.2, 2.0, -1.0, 0.2, 1.5, -0.3, 0.4]])
h = np.zeros((1, len(wi))); hs = []
for t in range(10):
    h = np.maximum(h @ wh.T + xx[:, t:t + 1] * wi, 0); hs.append(h[0].copy())
hs = np.array(hs)
print("scores", np.round(hs[-1] @ wo.T, 3))
for k in np.flatnonzero(np.abs(wo[7]) > 1e-12):
    print(f"  pos7 reads {name[k]:7s} w {wo[7, k]:+.4g}  value {hs[-1, k]:.4g}  term {wo[7, k] * hs[-1, k]:+.4g}")
for nm in ("h7", "c1", "c6", "c7", "keep", "mem0"):
    print(f"{nm:6s}", np.round(hs[:, idx[nm]], 3))
from algzoo.handbuilt_w12b import build_w12b
wi2, wh2, wo2, _, _ = build_w12b()
h = np.zeros((1, len(wi))); hs2 = []
for t in range(10):
    h = np.maximum(h @ wh2.T + xx[:, t:t + 1] * wi2, 0); hs2.append(h[0].copy())
hs2 = np.array(hs2)
print("w12b scores", np.round(hs2[-1] @ wo2.T, 3))
print("w12b mem0", np.round(hs2[:, idx["mem0"]], 3))
for k in np.flatnonzero(np.abs(wo2[7]) > 1e-12):
    print(f"  w12b pos7 reads {name[k]:7s} w {wo2[7, k]:+.4g}  value {hs2[-1, k]:.4g}  term {wo2[7, k] * hs2[-1, k]:+.4g}")
print("mem0 row w12b:", {name[k]: round(wh2[idx['mem0'], k], 6) for k in np.flatnonzero(np.abs(wh2[idx['mem0']]) > 1e-12)})
print("mem0 row w13b:", {name[k]: round(wh[idx['mem0'], k], 6) for k in np.flatnonzero(np.abs(wh[idx['mem0']]) > 1e-12)})
