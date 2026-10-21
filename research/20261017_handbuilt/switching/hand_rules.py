"""Step 3c: hand rules read off look_cells.log for neurons 0, 10, 11, 12, 13, 14; neurons 3 and 15 on order cells
(step, ages of the three largest; fitted, seed 32). Test seed 31, closed loop.
a1, a2: ages of the largest and second-largest positive numbers so far (-1 none); r = age of the largest of the
earlier numbers = a2 if the current number is a new largest (a1 == 0) else a1.
Step 0: off iff the input weight times x_0 <= 0 (exact). No positive number yet: neuron 0 off at step 1, rest on."""
import numpy as np
from swlib import *

X = lists(TEST_SEED, 200_000); K = kinds(X)
S0, P0, H0 = run(X); base = score(X, S0, S0, K); true_off = P0[:, :, MEM] <= 0
Xf = lists(FIT_SEED, 200_000); _, Pf, _ = run(Xf); off_f = Pf[:, :, MEM] <= 0
ag, _ = order_keys(X); agf, _ = order_keys(Xf)


def hand(ag, X):
    n = len(X); off = np.zeros((n, 10, 8), bool); a1, a2 = ag[:, :, 0], ag[:, :, 1]
    r = np.where(a1 == 0, a2, a1); col = {nn: j for j, nn in enumerate(MEM)}
    off[:, :, col[0]] = r == 2
    off[:, :, col[10]] = (r == 6) | ((a1 == 7) & (a2 == 6))
    off[:, :, col[11]] = r == 5
    off[:, :, col[13]] = (a1 == 0) & (a2 == -1)
    off[:, :, col[14]] = ((a1 == 0) & np.isin(a2, [-1, 2])) | ((a1 == 2) & (a2 == 0))
    none = a1 == -1
    off[none] = False; off[:, 1, col[0]] |= none[:, 1]
    off[:, 0] = (X[:, :1] * WIM[None, :]) <= 0
    return off


def cells(level, cols):
    pred = np.zeros((len(X), 10, 8), bool)
    for t in range(10):
        for l in range(1, level + 1):
            kf = agf[:, t, :l] @ (13 ** np.arange(l)); kt = ag[:, t, :l] @ (13 ** np.arange(l))
            u, inv, cnt = np.unique(kf, return_inverse=True, return_counts=True)
            mean = np.zeros((len(u), 8)); np.add.at(mean, inv, off_f[:, t]); mean /= cnt[:, None]
            pos = np.minimum(np.searchsorted(u, kt), len(u) - 1); ok = (u[pos] == kt) & (cnt[pos] >= 30)
            pred[ok, t] = mean[pos[ok]] > 0.5
    return pred


H = hand(ag, X); Hf = hand(agf, Xf)
print("hand rules, on/off agreement on the fitting lists: " + " ".join(f"{n}:{np.mean(Hf[:, :, j] == off_f[:, :, j]):.3f}" for j, n in enumerate(MEM)))
acc = lambda S: 100 * (score(X, S0, S)["all"][0] - base["all"][0])
print("one neuron by its hand rule, the rest by their own sign (points): " + " ".join(f"{n}:{acc(machine(X, lambda t, c: H[:, t], use=[n])[0]):+.2f}" for n in [0, 10, 11, 12, 13, 14]))
simple = [0, 10, 11, 12, 13, 14]
S, offs = machine(X, lambda t, c: H[:, t], use=simple)
report("hand rules for 0, 10-14; 3 and 15 by their own sign", X, S0, S, offs, true_off, base)
for lev, nm in ((2, "two ages"), (3, "three ages")):
    C = cells(lev, None); M = np.where(np.isin(MEM, simple)[None, None, :], H, C)
    S, offs = machine(X, lambda t, c: M[:, t]); report(f"hand rules for 0, 10-14; 3 and 15 on cells of {nm}", X, S0, S, offs, true_off, base, K if lev == 3 else None)
