"""Does the trained network use switching where the twin uses large levels? Per network: share of (list, step, neuron)
active, share of steps in the most common on/off pattern (steps 3-9), number of patterns covering 90% of steps, and
per neuron the share of steps it is on (steps 3-9). 50,000 lists."""
import numpy as np
from collections import Counter
import simlib
from algzoo.handbuilt_v18 import build_v18

X = simlib.lists(1212, 50_000)
for name, P in (("trained", (simlib.WI, simlib.WH, simlib.WO)), ("twin", tuple(np.asarray(p, float) for p in build_v18().params)),
                ("twin gains 30", tuple(np.asarray(p, float) for p in build_v18(g=30, w=30, a=20).params))):
    H = simlib.run(X, params=P)[2][:, 3:] > 0
    pats = Counter(map(bytes, np.packbits(H.reshape(-1, 16), axis=1)))
    cnt = np.array(sorted(pats.values(), reverse=True)); cum = np.cumsum(cnt) / cnt.sum()
    print(f"{name:14s}: active {H.mean():.2f}; most common pattern {cum[0]:.2f} of steps; patterns for 90% of steps {np.searchsorted(cum, 0.9) + 1}; "
          f"neurons on at 5-95% of steps {int(((H.mean((0, 1)) > 0.05) & (H.mean((0, 1)) < 0.95)).sum())}")
    print("   on-share per neuron: " + " ".join(f"{v:.2f}" for v in H.mean((0, 1))))
