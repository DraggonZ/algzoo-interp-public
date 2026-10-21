"""Reference for partial_scores.log: the vote per unit of numbers that do NOT partly switch the four (all four stay on)
at K = 1 and K >= 2, kept by the gate (at least 0.75 of the second-largest so far), split by their final role.
Vote per unit = change of the final score at the number's position minus the mean change elsewhere, when it is nudged.
300,000 lists, seed 7 (as partial_scores.py)."""
import numpy as np
from common import WI, WH, WO, C, lists, run
from rulelib import list_vars
X = lists(7, 300_000); B = len(X); o = np.argsort(-X, 1); lar, sec, thi = o[:, 0], o[:, 1], o[:, 2]
L0, P0, H0 = run(X); V = list_vars(X)
K = np.arange(10)[None] - V["R"]
allon = (P0[:, :, C] > 0).all(2); kept = X >= 0.75 * V["M2"]
rng = np.random.default_rng(2)
for lab, km in (("K = 1", K == 1), ("K >= 2", K >= 2)):
    m = km & allon & kept & (X > 0) & (V["R"] >= 0)
    has = m.any(1); j = np.argmax(m, 1); ii = np.where(has)[0]; ii = ii[rng.permutation(len(ii))[:60000]]; jj = j[ii]
    Xs = X[ii].copy(); a = np.arange(len(ii)); X2 = Xs.copy(); X2[a, jj] += 1e-3
    d = (run(X2)[0] - L0[ii]) / 1e-3; vote = d[a, jj] - (d.sum(1) - d[a, jj]) / 9
    for rl, rm in (("the answer", sec[ii] == jj), ("the third", thi[ii] == jj), ("fourth or lower", (sec[ii] != jj) & (thi[ii] != jj) & (lar[ii] != jj))):
        print(f"{lab}, kept, all four stay on, number is {rl:15s} ({rm.mean():.2f}): vote per unit median {np.median(vote[rm]):6.1f}")
