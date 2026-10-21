"""Step 3b: which neurons need more than the largest's age? For each neuron alone (others by their own sign), and for
groups, compare: (i) step-by-age table, (ii) cells of (step, three ages) as in order_cells.py. Also: all eight on
cells, except a group on the plain age table. Fits on seed 32, test on seed 31, closed loop."""
import numpy as np
from swlib import *

X = lists(TEST_SEED, 200_000); K = kinds(X)
S0, P0, H0 = run(X); base = score(X, S0, S0, K); true_off = P0[:, :, MEM] <= 0
Xf = lists(FIT_SEED, 200_000); _, Pf, _ = run(Xf); off_f = Pf[:, :, MEM] <= 0
ag, _ = order_keys(X); agf, _ = order_keys(Xf)


def cells(level):
    """prediction (B,10,8) from majority cells keyed by step and the first `level` ages, falling back to coarser keys"""
    pred = np.zeros((len(X), 10, 8), bool)
    for t in range(10):
        for l in range(1, level + 1):
            kf = agf[:, t, :l] @ (13 ** np.arange(l)); kt = ag[:, t, :l] @ (13 ** np.arange(l))
            u, inv, cnt = np.unique(kf, return_inverse=True, return_counts=True)
            mean = np.zeros((len(u), 8)); np.add.at(mean, inv, off_f[:, t]); mean /= cnt[:, None]
            pos = np.searchsorted(u, kt); pos = np.minimum(pos, len(u) - 1); ok = (u[pos] == kt) & (cnt[pos] >= 30)
            pred[ok, t] = mean[pos[ok]] > 0.5
    return pred


P1, P2, P3 = cells(1), cells(2), cells(3)
acc = lambda S: 100 * (score(X, S0, S)["all"][0] - base["all"][0])
print("one neuron by the rule, the rest by their own sign (change in points):")
print("  neuron | age table | + second's age | + third's age")
for n in MEM:
    print(f"  {n:6d} | " + " | ".join(f"{acc(machine(X, lambda t, c, P=P: P[:, t], use=[n])[0]):+9.2f}" for P in (P1, P2, P3)))
mix = lambda simple, P: (lambda t, c: np.where(np.isin(MEM, simple)[None, :], P1[:, t], P[:, t]))
for simple in [[0, 10, 11, 12, 13, 14], [10, 11, 12, 13], [10, 11, 12], [0, 10, 11]]:
    for P, nm in ((P2, "two ages"), (P3, "three ages")):
        S, offs = machine(X, mix(simple, P)); report(f"{simple} on the age table, the rest on cells of {nm}", X, S0, S, offs, true_off, base)
