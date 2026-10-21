"""Does ARC's neuron 5 sharpen the revealed vote? Lists: old largest 1.0 at p, bigger 2.0 at q = p + 2 (fillers
-0.5, and random small fillers). Revealed vote = scores(list) - scores(same list without the 1.0). Variant: at the
step after the reveal (q + 1), neuron 5's input to the timer neurons is removed (in both runs). Sharpness: the
vote at p minus the largest vote at any other position 0-8 (positive = p wins on its own)."""
import numpy as np
from algzoo.core import example

wi, wh, wo = (np.asarray(w, dtype=float) for w in example()); wi = wi.reshape(-1)
tim = [0, 3, 9, 10, 11, 12, 13, 14, 15]


def scores(x, block_step=None):
    h = np.zeros(16)
    for t, v in enumerate(x):
        W = wh.copy()
        if t == block_step: W[tim, 5] = 0.0
        h = np.maximum(W @ h + wi * v, 0)
    s = wo @ h; return s - s.mean()


np.set_printoptions(precision=0, suppress=True, linewidth=150)
rng = np.random.default_rng(20)
for p in (1, 3, 5):
    q = p + 2
    for kind in ("fillers -0.5", "random fillers"):
        res = {False: [], True: []}
        prof = {}
        for rep in range(1 if kind == "fillers -0.5" else 200):
            x = np.full(10, -0.5) if kind == "fillers -0.5" else np.minimum(rng.standard_normal(10) * 0.5 - 0.3, 0.6)
            x[p], x[q] = 1.0, 2.0
            x0 = x.copy(); x0[p] = -0.5
            for blk in (False, True):
                d = scores(x, q + 1 if blk else None) - scores(x0, q + 1 if blk else None)
                res[blk].append(d[p] - np.delete(d[:9], p).max())
                prof.setdefault(blk, d)
        print(f"p={p} q={q} {kind:15s} margin of p over the best other position: normal {np.mean(res[False]):6.1f}, "
              f"neuron 5 blocked {np.mean(res[True]):6.1f}")
        if kind == "fillers -0.5":
            print("     normal ", prof[False]); print("     blocked", prof[True])
