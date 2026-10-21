"""Per-neuron results for v18 under the active-set fit, with the menu plus v18's variables, and with a menu that also
holds v18's memory variables (every positive number by age, not only numbers kept by the trained network's rule)."""
import numpy as np
from algzoo.handbuilt_v18 import build_v18
import simlib
from simlib import lists, task_vars
from sim_groups import lag
from twin_rightvars import feats_plus
from fit_active import fit_active


def feats_v18mem(X):
    V = task_vars(X); x = V["x"]; pos = np.maximum(x, 0)
    return np.concatenate([feats_plus(X), np.stack([pos] + [lag(pos, a) for a in range(1, 10)], -1)], -1)


Xf, Xs = lists(101, 50_000), lists(202, 50_000); y = Xs.argsort(1)[:, -2]
P = tuple(np.asarray(p, float) for p in build_v18().params)
_, _, Hf = simlib.run(Xf, params=P); Ls, _, Hs = simlib.run(Xs, params=P); base = (Ls.argmax(1) == y).mean()
names = ["store", "detect", "keep", "flip", "clock0", "clock1", "clock2", "flag"] + [f"mem{j}" for j in range(8)]
for label, fn in (("menu plus v18's clock variables", feats_plus), ("plus v18's memory variables too", feats_v18mem)):
    Ff, Fs = fn(Xf), fn(Xs); sims = {}; row = []
    for n in range(16):
        co = fit_active(Ff, Hf[:, :, n]); sims[n] = np.maximum(Fs @ co, 0)
        row.append((simlib.run(Xs, {n: sims[n]}, params=P)[0].argmax(1) == y).mean())
    al = (simlib.run(Xs, sims, params=P)[0].argmax(1) == y).mean()
    print(f"{label}: base {base:.3f}; all 16 {al:.3f}; " + ", ".join(f"{nm} {r:.3f}" for nm, r in zip(names, row)), flush=True)
