"""Where does v19 (trained scale, kappa 0.5) lose accuracy? Variants of the forward run, timer = clock 4-6 and flip 3:
  as built;
  no clipping from kept numbers: the timer's ReLU is applied only at record steps (x_t > m1), identity otherwise;
  perfect wipe: at record steps the timer is set to exactly zero;
  both.
Compared with kappa 0. 100,000 lists."""
import numpy as np
from algzoo.handbuilt_v19 import build_v19
import simlib

X = simlib.lists(77, 100_000); y = X.argsort(1)[:, -2]; T = [3, 4, 5, 6]
cm = np.maximum.accumulate(X, 1); prev = np.c_[np.zeros((len(X), 1)), cm[:, :-1]]; rec = X > np.maximum(prev, 0)
def run(P, noclip, perfect):
    h = np.zeros((len(X), 16))
    for t in range(10):
        z = h @ P[1].T + X[:, t:t + 1] * P[0]; hn = np.maximum(z, 0)
        r = rec[:, t]
        if noclip:
            hn[np.ix_(~r, T)] = z[np.ix_(~r, T)]
        if perfect:
            hn[np.ix_(r, T)] = 0.0
        h = hn
    return ((h @ P[2].T).argmax(1) == y).mean()
for kappa in (0.0, 0.5):
    P = tuple(np.asarray(p, float) for p in build_v19(g=10, a=10, w=10, lam=10.0, c=-9.0, eps=0.1, kappa=kappa).params)
    print(f"kappa {kappa}: as built {run(P, False, False):.4f}; no clipping from kept numbers {run(P, True, False):.4f}; perfect wipe {run(P, False, True):.4f}; both {run(P, True, True):.4f}", flush=True)
