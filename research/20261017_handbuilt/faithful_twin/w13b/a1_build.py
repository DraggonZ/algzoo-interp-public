"""w13b a1: own accuracy of the no-backward-wave build (handbuilt_w13b), against w12b, by the largest's position.
Also checks on one list that J and the clocks switch on at the ages the design says."""
import numpy as np
from algzoo.handbuilt_w12b import build_w12b
from algzoo.handbuilt_w13b import build_w13b

rng = np.random.default_rng(0)
x = rng.standard_normal((400000, 10))
sec = np.argsort(x, 1)[:, -2]; top = np.argmax(x, 1)


def run(wi, wh, wo, xx=x):
    h = np.zeros((len(xx), len(wi))); hs = []
    for t in range(10):
        h = np.maximum(h @ wh.T + xx[:, t:t + 1] * wi, 0); hs.append(h)
    return np.argmax(h @ wo.T, 1), np.stack(hs, 1)


var = {"w12b": build_w12b()[:3]}
b = build_w13b(report=True); var["w13b"] = b[:3]
var["w13b keep pos4 read"] = build_w13b(drop_pos4=False)[:3]
var["w13b drop all pos4"] = build_w13b(drop_pos4="all")[:3]
for nm, W in var.items():
    p, _ = run(*W); ok = p == sec
    print(f"{nm:22s} acc {ok.mean()*100:6.2f}  names largest {np.mean(p == top)*100:5.2f}  by largest's position:",
          " ".join(f"{ok[top == j].mean()*100:5.1f}" for j in range(10)))
idx = b[3]
xx = np.array([[0.5, 0.2, -0.3, 2.0, -1.0, -0.5, 0.1, -1.2, 0.3, -0.4]])
_, hs = run(*b[:3], xx)
print("one list, record at 3 (d = 1.5):")
print("  J-c7   ", np.round(hs[0, :, idx["linear"]] - hs[0, :, idx["c7"]], 3))
for nm in ("late", "c7"):
    print(f"  {nm:7s}", np.round(hs[0, :, idx[nm]], 3))
for nm in ("clock0", "clock1", "clock2", "turn0", "turn1", "turn2"):
    print(f"  {nm:7s}-c7", np.round(hs[0, :, idx[nm]] - hs[0, :, idx['c7']], 3))
print("  scores", np.round(hs[0, -1] @ b[2].T, 3))
p, _ = run(*b[:3]); ok = p == sec
for j in (4, 5, 8):
    m = top == j
    print(f"largest at {j}: accuracy by second's position", " ".join(f"{ok[m & (sec == k)].mean()*100:5.1f}" for k in range(10)),
          "| named:", np.bincount(p[m & ~ok], minlength=10))
