"""Follow-up to vote_in_cancellers.py: is the stored vote due to partial switching? z one step after a record, record
still largest at the end; split by whether some but not all of the four cancelling neurons are off at z's step."""
import numpy as np
import simlib

X = simlib.lists(777, 200_000); B = len(X); C = [1, 6, 7, 8]; rest = [n for n in range(16) if n not in C]
_, P, H = simlib.run(X)
cm = np.maximum.accumulate(X, 1); prevmax = np.c_[np.full((B, 1), -np.inf), cm[:, :-1]]; rec = X > prevmax
out = {True: {"c": [], "r": []}, False: {"c": [], "r": []}}
for p in range(1, 7):
    zp = p + 1; R = X[:, p]; z = X[:, zp]
    m2 = np.sort(np.c_[np.zeros((B, 2)), X[:, :zp]], 1)[:, -2]
    ok = rec[:, p] & (cm[:, p] == R) & (z > 0) & (z < R) & (z >= 0.75 * m2) & (cm[:, -1] == R)
    off = (P[:, zp][:, C] <= 0).sum(1); partial = (off > 0) & (off < 4)
    for flag in (True, False):
        idx = np.flatnonzero(ok & (partial == flag))[:4000]
        if len(idx) < 30: continue
        Xn = X[idx].copy(); Xn[:, zp] += 0.01; _, _, Hn = simlib.run(Xn)
        dh = (Hn[:, -1] - H[idx, -1]) / 0.01; w = simlib.WO[zp]
        out[flag]["c"] += list(dh[:, C] @ w[C]); out[flag]["r"] += list(dh[:, rest] @ w[rest])
for flag in (True, False):
    print(f"z one step after a record, cancellers {'partly switched' if flag else 'not partly switched'} at z's step: via cancelling neurons {np.median(out[flag]['c']):6.1f}, via the rest {np.median(out[flag]['r']):6.1f} ({len(out[flag]['c'])} lists)")
