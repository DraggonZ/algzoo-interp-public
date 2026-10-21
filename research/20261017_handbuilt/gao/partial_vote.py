"""Does the cancellers' partial clipping shape a kept number's own vote? For lists where position s (2-6) holds a kept,
non-record positive number that partly switches the cancellers at its arrival step, nudge it by 0.01 and measure its
vote (change in its position's final score per unit) with the network as is, and with the cancellers left unclipped
at that one step. Also the vote's change on the second-largest's position for lists where s holds the second."""
import numpy as np
import simlib
from simlib import task_vars

X = simlib.lists(2727, 200_000); B = len(X); r = np.arange(B); C = [1, 6, 7, 8]; V = task_vars(X)


def final(Xs, s, unclip):
    h = np.zeros((len(Xs), 16))
    for t in range(10):
        pre = h @ simlib.WH.T + Xs[:, t:t + 1] * simlib.WI; h = np.maximum(pre, 0)
        if unclip and t == s:
            off = (pre[:, C] <= 0).sum(1); m = (off > 0) & (off < 4); h[np.ix_(m, C)] = pre[np.ix_(m, C)]
    return h @ simlib.WO.T


for s in range(2, 7):
    _, PRE, _ = simlib.run(X); off = (PRE[:, s][:, C] <= 0).sum(1)
    sel = (V["x"][:, s] > 0) & (V["newmax"][:, s] == 0) & (V["x"][:, s] >= 0.75 * V["m2"][:, s]) & (off > 0) & (off < 4)
    Xs = X[sel]; X2 = Xs.copy(); X2[:, s] += 0.01
    for unclip in (False, True):
        v = (final(X2, s, unclip)[:, s] - final(Xs, s, unclip)[:, s]) / 0.01
        print(f"position {s}, kept number partly switching the cancellers ({sel.sum()} lists), {'unclipped at its step' if unclip else 'as is':22s}: its vote per unit, median {np.median(v):6.1f}", flush=True)

# Compare: the vote per unit of an overtaken record (a number that was the largest when it arrived and is the
# second-largest of the list) at the same positions, as is.
m1p = np.stack([np.max(np.c_[np.full(B, -np.inf), X[:, :t]], 1) for t in range(10)], 1); y = X.argsort(1)[:, -2]
for s in range(2, 7):
    sel = (y == s) & (X[r, s] > m1p[r, s]) & (X[r, s] > 0)
    Xs = X[sel]; X2 = Xs.copy(); X2[:, s] += 0.01
    v = (final(X2, s, False)[:, s] - final(Xs, s, False)[:, s]) / 0.01
    print(f"position {s}, overtaken record that is the second-largest ({sel.sum()} lists): its vote per unit, median {np.median(v):6.1f}", flush=True)
