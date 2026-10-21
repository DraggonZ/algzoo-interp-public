"""Exact properties of v18 on hand-made lists (all other numbers -0.5, so they are forgotten... here: kept at 0)."""
import numpy as np, sys
sys.path.insert(0, "scripts")
from handbuilt_v18 import build_v18, scores
np.set_printoptions(precision=4, suppress=True, linewidth=160)
net = build_v18()
worst_lone = worst_rev = 0.0
for r in range(10):
    for v in (0.3, 1.0, 2.5):
        x = np.full(10, -0.5); x[r] = v
        worst_lone = max(worst_lone, np.abs(scores(net, x[None])[0][:9]).max())
        for L in range(r + 1, 10):
            for u in (1.01, 1.5):
                x2 = x.copy(); x2[L] = v * u
                sc = scores(net, x2[None])[0]
                if r <= 8: worst_rev = max(worst_rev, abs(sc[r] - 0.25 * v))
print(f"lone record: largest |score| at positions 0-8 {worst_lone:.2e} (position 9 holds the last number's own vote)")
print(f"overtaken record: largest |score at its position - 0.25 x value| {worst_rev:.2e}")
for vals in ({3: 1.0, 6: 1.5}, {3: 1.0, 4: 1.5}, {3: 1.0, 4: 0.9, 6: 1.5}, {1: 0.6, 3: 1.0, 6: 1.5, 7: 1.2}):
    x = np.full(10, -0.5)
    for k, v in vals.items(): x[k] = v
    s = scores(net, x[None])[0]; print(f"{str(vals):34s}", s, "->", s.argmax())
