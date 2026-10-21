"""Why does first order miss group substitutions by about 13 points (first_order_accounts.log)? If the miss is switching,
averaging the backward direction of the true run and of the substituted run (trapezoid rule; in the substituted run the
substituted neurons are forced, so no gradient passes back through them) should close much of it. Same accounts as
first_order_accounts.py, substituted only. Lead measured between the true winner and runner-up."""
import numpy as np
from scipy.stats import spearmanr
import simlib
from simlib import run
from matched_size import X, y, L, H, G, MIS, mis, account_features, fitted

r = np.arange(len(X)); k = L.argmax(1); j = np.argsort(L, 1)[:, -2]; lead = L[r, k] - L[r, j]
def back(P, forced=()):
    g = simlib.WO[k] - simlib.WO[j]; Gt = np.zeros((len(X), 10, 16))
    for t in range(9, -1, -1):
        Gt[:, t] = g; m = (P[:, t] > 0).astype(float); m[:, list(forced)] = 0; g = (g * m) @ simlib.WH
    return Gt
G0 = back(run(X)[1]); rows = []
for gname, ns in G.items():
    accounts = [("ours", "account", fitted(gname, account_features))] + [("misreading", kk, fitted(gname, lambda Xs, n, kk=kk: mis(Xs, n, kk))) for kk in MIS[gname]]
    for kind, label, sims in accounts:
        Ls, Ps, _ = run(X, sims); meas = (Ls.argmax(1) == y).mean()
        G1 = back(Ps, ns); err = {n: sims[n] - H[:, :, n] for n in ns}
        d0 = sum((G0[:, :, n] * e).sum(1) for n, e in err.items()); d1 = sum((0.5 * (G0[:, :, n] + G1[:, :, n]) * e).sum(1) for n, e in err.items())
        p0 = np.where(lead + d0 < 0, j == y, k == y).mean(); p1 = np.where(lead + d1 < 0, j == y, k == y).mean()
        rows.append((meas, p0, p1))
        print(f"{gname:15s} {kind:10s} {label:20s}: measured {meas:.3f}; first order {p0:.3f}; trapezoid {p1:.3f}", flush=True)
R = np.array(rows)
for c, lab in ((1, "first order"), (2, "trapezoid")):
    print(f"{lab}: rank correlation {spearmanr(R[:, 0], R[:, c])[0]:.2f}; median |miss| {np.median(np.abs(R[:, 0] - R[:, c])):.3f}")
