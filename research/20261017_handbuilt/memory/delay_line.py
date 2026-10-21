"""Question: does a vote travel along the memory neurons like a delay line (12 -> 11 -> 10 ...), as the vote readout
(vote_by_neuron.log) and the switching cascade (offrate_age.log: 12, 11, 10 off at ages 4, 5, 6) suggest?
Per-unit change of each memory neuron's activity at each age (steps since the nudged number arrived), mean:
(a) a kept number at s = 1 or 2 that ends as the second-largest, after the largest (nudge x_s);
(b) a record at r = 1 or 2 that stays the largest (nudge x_r).
For each neuron the age of its largest |change| is marked. 1,000,000 lists, seed 21."""
import numpy as np
from memlib import MEM, run, lists
X = lists(21, 1_000_000); o = X.argsort(1); eps = 1e-4
pm = np.maximum.accumulate(np.c_[np.zeros(len(X)), X], 1)[:, :-1]
for title, s, sel in [("(a) kept number at 1", 1, (o[:, -2] == 1) & (o[:, -1] < 1) & (X[:, 1] > 0.3)),
                      ("(a) kept number at 2", 2, (o[:, -2] == 2) & (o[:, -1] < 2) & (X[:, 2] > 0.3)),
                      ("(b) record at 1, stays largest", 1, (X[:, 1] > pm[:, 1] + 0.05) & (X[:, 1] > 0.3) & (X[:, 2:].max(1) < X[:, 1] - 0.05)),
                      ("(b) record at 2, stays largest", 2, (X[:, 2] > pm[:, 2] + 0.05) & (X[:, 2] > 0.3) & (X[:, 3:].max(1) < X[:, 2] - 0.05))]:
    Xs = X[sel][:100_000]; dX = np.zeros_like(Xs); dX[:, s] = eps
    d = ((run(Xs + dX)[2] - run(Xs)[2]) / eps).mean(0)       # (10 steps, 16)
    print(f"{title} ({len(Xs)} lists): per-unit change by age 0..{9 - s} (rows = memory neurons); * = largest |change|")
    for n in MEM:
        v = d[s:, n]; k = int(np.argmax(np.abs(v)))
        print(f"   neuron {n:2d}: " + " ".join(f"{x:6.1f}{'*' if i == k else ' '}" for i, x in enumerate(v)))
