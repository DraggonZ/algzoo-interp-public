"""ARC: where is the old largest's position held k steps after the wipe? Copy neuron groups from a donor
(smaller number at a different position) at step p1 + k. Lists: smaller 1.0 at a, bigger 1.5 at p1, zeros."""
import itertools

import numpy as np

from algzoo.core import example

wi, wh, wo = example()
L = [1, 6, 7, 8]; R = [k for k in range(16) if k not in L]


def run(x, patch=None):
    h = np.zeros(16); hs = []
    for t, v in enumerate(x):
        h = np.maximum(0, wh @ h + wi * v)
        if patch and patch[0] == t:
            h = h.copy(); h[patch[1]] = patch[2][t][patch[1]]
        hs.append(h)
    return (wo @ h).argmax(), hs


for k in (0, 1, 2, 4):
    res = {"other twelve": [], "neurons 1, 6, 7, 8": [], "all 16": []}
    for a, a2, p1 in itertools.permutations(range(10), 3):
        if p1 <= max(a, a2) or p1 + k > 9:
            continue
        x = np.zeros(10); x[a] = 1.0; x[p1] = 1.5; d = np.zeros(10); d[a2] = 1.0; d[p1] = 1.5
        _, hd = run(d)
        for lab, G in (("other twelve", R), ("neurons 1, 6, 7, 8", L), ("all 16", list(range(16)))):
            ans, _ = run(x, (p1 + k, G, hd)); res[lab].append(ans == a2)
    print(f"{k} step(s) after the wipe: answer moves to the donor's position when copying " + ", ".join(f"{lab}: {np.mean(v):.2f}" for lab, v in res.items()) + f"  (n={len(res['all 16'])})")
