"""Question: how much of a vote at each position is read through neurons 3 and 15? Nudge the final second-largest
(per unit, mean): (a) a kept number at s after the largest; (b) an overtaken record at r (nudged alone). The change
of the score at its own position, split by the neuron it is read through (final activity change x readout weight).
Columns: position 0-8 of the second-largest. 1,000,000 lists, seed 21."""
import numpy as np
from memlib import MEM, WO, run, lists
X = lists(21, 1_000_000); o = X.argsort(1); eps = 1e-4
pm = np.maximum.accumulate(np.c_[np.zeros(len(X)), X], 1)[:, :-1]; rec = (X > pm) & (X > 0)
for title, sel_fn in [("(a) kept number after the largest", lambda s: (o[:, -2] == s) & (o[:, -1] < s) & (X[:, s] > 0.3)),
                      ("(b) overtaken record", lambda s: (o[:, -2] == s) & (o[:, -1] > s) & rec[:, s] & (X[:, s] > 0.3))]:
    rows = {n: [] for n in MEM}; rows["memory"] = []; rows["others"] = []; rows["total"] = []
    for s in range(9):
        Xs = X[sel_fn(s)][:100_000]
        if len(Xs) < 500:
            for k in rows: rows[k].append(np.nan)
            continue
        dX = np.zeros_like(Xs); dX[:, s] = eps
        _, _, H0 = run(Xs); _, _, H1 = run(Xs + dX); dh = ((H1 - H0)[:, 9] / eps).mean(0)
        part = dh * WO[s]
        for n in MEM: rows[n].append(part[n])
        rows["memory"].append(part[MEM].sum()); rows["others"].append(part.sum() - part[MEM].sum()); rows["total"].append(part.sum())
    print(f"{title}: per-unit change of the score at its own position, read through each neuron; columns = position 0-8")
    for k, v in rows.items(): print(f"   {str(k):8s} " + " ".join(f"{x:6.1f}" for x in v))
