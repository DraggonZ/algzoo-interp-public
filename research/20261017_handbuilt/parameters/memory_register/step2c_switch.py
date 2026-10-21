"""What switches memory neurons off? For each memory neuron and step, compare the off rate by the age of the current
largest (steps since it arrived) and by step; and the neuron's input split by source position (exact split)."""
import numpy as np
from decoders import *
X = np.random.default_rng(5).standard_normal((30000, 10))
H, J = contributions(X); kind, over = labels(X, H)
runarg = np.array([[np.argmax(x[:t]) for t in range(1, 11)] for x in X])  # position of largest after t numbers
np.set_printoptions(precision=2, suppress=True, linewidth=200)
for n in [0, 3, 15, 10, 11, 14]:
    print(f"\n== neuron {n}: off rate by step (columns 3..10) and by the largest's age at that step (rows 0..6)")
    tab = np.full((7, 8), np.nan)
    for t in range(3, 11):
        age = (t - 1) - runarg[:, t - 1]
        for a in range(7):
            m = age == a
            if m.sum() > 300: tab[a, t - 3] = (H[m, t, n] <= 0).mean()
    print(tab)
    # input to neuron n at step t split by source: which positions' contributions push it below zero?
    t = 6
    pre = np.einsum("j,njp->np", wh[n], J[:, t - 1]) ; pre[:, t - 1] += wi[n]  # input at step t from x_p, per unit x_p
    contrib = pre * X
    offm = H[:, t, n] <= 0
    age = (t - 1) - runarg[:, t - 1]
    # mean contribution by role: the current largest, kept numbers, others
    lg = np.zeros_like(contrib, bool); lg[np.arange(len(X)), runarg[:, t - 1]] = True
    kp = (kind == 1); kp[:, t:] = False
    for nm, m in [("off", offm), ("on", ~offm)]:
        print(f"   step {t}, lists with neuron {n} {nm} ({m.mean():.2f}): input from the largest {np.where(lg, contrib, 0)[m].sum(1).mean():6.2f},"
              f" from kept numbers {np.where(kp & ~lg, contrib, 0)[m].sum(1).mean():6.2f}, from the rest {np.where(~kp & ~lg, contrib, 0)[m].sum(1).mean():6.2f}")
