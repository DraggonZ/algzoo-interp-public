"""When a gate's pushes are removed, what does the network name instead? (200,000 lists, seed 13 as g13_n5_job.py.)
Keep gate: pushes removed at the steps after records, or after other numbers. Late gate: push onto the hiding
neurons removed (all steps). Broken lists = right when intact, wrong after the removal."""
import numpy as np
from gatelib import WI, WH, WO, CANC, lists, run, task_vars, answer

X = lists(13); V = task_vars(X); y = answer(X); L0 = run(X)[0]; ok0 = L0.argmax(1) == y
rec_next = np.zeros((len(X), 10), bool); rec_next[:, 1:] = V["new"][:, :9].astype(bool)
o = X.argsort(1); iL, iS, iT = o[:, -1], o[:, -2], o[:, -3]


def removed(when):
    def hook(t, hp, pre, h):
        if t == 0:
            return h
        m = when[:, t - 1]
        corr = -np.outer(hp[:, 5] * m, WH[:, 5])
        return np.maximum(pre + corr, 0)
    return run(X, hook=hook)[0]


wh = WH.copy(); wh[CANC, 9] = 0
cases = {"keep gate silent after records": removed(rec_next), "keep gate silent after other numbers": removed(~rec_next),
         "late gate's push onto hiding neurons removed": run(X, params=(WI, wh, WO))[0]}
for name, L in cases.items():
    a = L.argmax(1); bad = ok0 & (a != y)
    for kind, m in (("S before L", iS < iL), ("S after L", iS > iL)):
        b = bad & m
        print(f"{name}, {kind}: {100 * b.sum() / m.sum():.1f}% of lists broken; they name the largest {100 * (a[b] == iL[b]).mean():.0f}%, "
              f"the third {100 * (a[b] == iT[b]).mean():.0f}%, the position after S {100 * (a[b] == iS[b] + 1).mean():.0f}%, "
              f"the position after L {100 * (a[b] == iL[b] + 1).mean():.0f}%, other {100 * ((a[b] != iL[b]) & (a[b] != iT[b]) & (a[b] != iS[b] + 1) & (a[b] != iL[b] + 1)).mean():.0f}%")
