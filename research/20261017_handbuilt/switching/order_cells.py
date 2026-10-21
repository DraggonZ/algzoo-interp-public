"""Step 2: how much do the ages and order of the numbers alone decide the switching? (upper bound for rules on order)
Each neuron's majority on/off in cells keyed by the step and the ages of the largest positive numbers so far, fitted
on seed 32 (200,000 lists), applied closed loop on seed 31. Cells with under 30 fitting lists fall back to the
coarser key. Not a compact rule: it bounds what any rule on order alone can do."""
import numpy as np
from swlib import *

X = lists(TEST_SEED, 200_000); K = kinds(X)
S0, P0, H0 = run(X); base = score(X, S0, S0, K); true_off = P0[:, :, MEM] <= 0
Xf = lists(FIT_SEED, 200_000); _, Pf, _ = run(Xf); off_f = Pf[:, :, MEM] <= 0
ag, rc = order_keys(X); agf, rcf = order_keys(Xf)


def keyfun(ages, isrec, X, t, level):
    cols = [np.full(len(ages), t)]
    if level >= 1: cols.append(ages[:, t, 0])
    if level >= 2: cols.append(ages[:, t, 1])
    if level >= 3: cols.append(isrec[:, t, 1])
    if level >= 4: cols.append(ages[:, t, 2])
    if level >= 5: cols.append(isrec[:, t, 2])
    if level >= 6: cols.append(X[:, t] > 0)
    return [tuple(r) for r in np.stack(cols, 1)]


LEVELS = {1: "step, age of largest", 2: "+ age of second", 3: "+ second was a record", 4: "+ age of third",
          5: "+ third was a record", 6: "+ current number positive"}
for L in LEVELS:
    tables = {}
    for l in range(1, L + 1):
        tab = {}
        for t in range(10):
            ks = keyfun(agf, rcf, Xf, t, l); d = {}
            for i, k in enumerate(ks): d.setdefault(k, []).append(i)
            for k, idx in d.items():
                if len(idx) >= 30: tab[k] = off_f[idx, t].mean(0) > 0.5
        tables[l] = tab
    pred = np.zeros((len(X), 10, 8), bool)
    for t in range(10):
        keys = {l: keyfun(ag, rc, X, t, l) for l in range(1, L + 1)}
        for i in range(len(X)):
            for l in range(L, 0, -1):
                v = tables[l].get(keys[l][i])
                if v is not None: pred[i, t] = v; break
    S, offs = machine(X, lambda t, c: pred[:, t])
    report(f"order cells: {LEVELS[L]} ({len(tables[L])} cells)", X, S0, S, offs, true_off, base, K if L in (2, 6) else None)
