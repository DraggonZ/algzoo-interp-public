"""Rotation speeds of the step matrix on the on/off patterns that occur on quiet steps after two inputs."""
from collections import Counter

import numpy as np

from algzoo.core import example

w_hi, w_hh, w_oh = example()
pats = Counter()
for r in (0.3, 0.7, 0.95):
    for p1 in range(10):
        for p2 in range(10):
            if p1 == p2:
                continue
            x = np.zeros(10); x[p1] = 1.0; x[p2] = r
            h = np.zeros(16)
            for t in range(10):
                pre = w_hh @ h + x[t] * w_hi
                h = np.maximum(0, pre)
                if t > max(p1, p2):
                    pats[tuple((pre > 0).astype(int))] += 1
print(f"{len(pats)} distinct on/off patterns on quiet steps; most common:")
for pat, c in pats.most_common(6):
    on = np.array(pat, dtype=float)
    M = on[:, None] * w_hh
    ev = np.linalg.eigvals(M)
    ev = ev[np.argsort(-np.abs(ev))]
    off = [i for i in range(16) if not pat[i]]
    desc = ", ".join(f"{abs(e):.2f}@{np.angle(e)/(2*np.pi):+.3f}" for e in ev[:10] if abs(e) > 0.05)
    print(f"  {c:4d} times, off neurons {off}: size@cycles-per-step: {desc}")
