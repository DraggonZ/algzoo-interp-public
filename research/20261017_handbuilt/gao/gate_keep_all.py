"""Does forgetting help ARC's network? Force the gate (neuron 5) to keep the numbers it forgets: one step after a
positive number below about 0.75 x the second-largest so far (not a record), set neuron 5 to the value it would have
for a kept number, then run on. Neuron 5's active value one step after a kept non-record number is fitted as
a*x + b*m2 (least squares on kept steps); for forgotten numbers we set it to s * a * x (s = 0.25, 0.5, 1: the share
of a kept number's vote they get; the m2 term, which is what silences them, is dropped). 200,000 lists."""
import numpy as np
import simlib
from simlib import lists, task_vars

X = lists(606, 200_000); y = X.argsort(1)[:, -2]; B = len(X)
L, PRE, H = simlib.run(X); base = (L.argmax(1) == y).mean()
V = task_vars(X); x, m2, m1 = V["x"], V["m2"], V["m1"]
rec = V["newmax"] > 0
kept = (x > 0) & ~rec & (x >= 0.75 * m2); forg = (x > 0) & ~rec & (x < 0.75 * m2)
# neuron 5 one step after step t
h5n = np.full_like(x, np.nan); h5n[:, :-1] = H[:, 1:, 5]
k = kept[:, :-1] & (h5n[:, :-1] > 0)
A = np.stack([x[:, :-1][k], m2[:, :-1][k]], 1); co = np.linalg.lstsq(A, h5n[:, :-1][k], rcond=None)[0]
r2 = 1 - ((A @ co - h5n[:, :-1][k]) ** 2).mean() / h5n[:, :-1][k].var()
print(f"neuron 5 one step after a kept number ~ {co[0]:.2f} x + {co[1]:.2f} m2 (R2 {r2:.3f}); base accuracy {base:.3f}")
print(f"forgotten numbers per list: {forg[:, :-1].sum(1).mean():.2f}")


def run_forced(s):
    h = np.zeros((B, 16))
    for t in range(10):
        h = np.maximum(h @ simlib.WH.T + X[:, t:t + 1] * simlib.WI, 0)
        if t:
            f = forg[:, t - 1]
            h[f, 5] = s * co[0] * X[f, t - 1]
    return h @ simlib.WO.T


for s in (0.25, 0.5, 1.0):
    Lf = run_forced(s); print(f"forgotten numbers kept with share {s}: accuracy {(Lf.argmax(1) == y).mean():.3f}", flush=True)
# control: the same forcing on as many randomly chosen kept non-record steps (adds to neuron 5 where it already fires)
