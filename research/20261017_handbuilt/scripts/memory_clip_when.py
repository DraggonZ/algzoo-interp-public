"""When do ARC's memory neurons clip? Around new largests, or anywhere?"""
import sys; sys.path.insert(0, '/data/output/src/algzoo/src')
import numpy as np, torch as th
from algzoo import core
np.set_printoptions(precision=2, suppress=True, linewidth=160)
p = core.example(); wi, wh, wo = p
X = th.randn(200_000, 10, generator=th.Generator().manual_seed(1), dtype=th.float64)
logits, H = core.forward(p, X, return_hidden=True); H = H.numpy(); X = X.numpy()
mem = [0, 3, 10, 11, 12, 13, 14, 15]
m1 = np.maximum.accumulate(np.maximum(X, 0), axis=1)
prev = np.concatenate([np.zeros((len(X), 1)), m1[:, :-1]], 1)
rec = (X > prev) & (X > 0)                               # input t is a new largest
off = H[:, 1:, :][:, :, mem] == 0                        # off[:, t] = state after input t
print("memory neurons", mem, ": share off, by what input t was (t = 2..9), for the state after t, t+1, t+2")
for lab, cond in (("new largest", rec), ("kept-size (0.75..1 of largest, not new)", (~rec) & (X > 0.75 * prev) & (X > 0)),
                  ("positive, below 0.75 of largest", (~rec) & (X > 0) & (X <= 0.75 * prev)), ("negative", X <= 0)):
    for lag in (0, 1, 2):
        c = cond[:, 2:10 - lag]; o = off[:, 2 + lag:10, :]
        print(f"  {lab if lag == 0 else '':42s} after t+{lag} ", o[c].mean(0), " mean", round(o[c].mean(), 3))
A = H[:, 1:, :][:, :, mem]
c = rec[:, 2:9]; print("mean activity, state after t+1: new largest at t / otherwise")
print("  ", A[:, 3:10][c].mean(0) / A[:, 3:10][~c].mean(0))
