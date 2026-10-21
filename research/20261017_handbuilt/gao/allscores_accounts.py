"""Is the 13-point first-order miss on group substitutions (first_order_accounts.log) because a third position can win?
First-order prediction of all ten final scores (one backward pass per position, true run's on/off pattern), predicted
answer = argmax. Same accounts, substituted only."""
import numpy as np
from scipy.stats import spearmanr
import simlib
from simlib import run
from matched_size import X, y, L, H, G, MIS, mis, account_features, fitted

_, P, _ = run(X); M = (P > 0).astype(float)
Gp = np.zeros((10, len(X), 10, 16))
for pos in range(10):
    g = np.broadcast_to(simlib.WO[pos], (len(X), 16)).copy()
    for t in range(9, -1, -1):
        Gp[pos, :, t] = g; g = (g * M[:, t]) @ simlib.WH
rows = []
for gname, ns in G.items():
    accounts = [("ours", "account", fitted(gname, account_features))] + [("misreading", kk, fitted(gname, lambda Xs, n, kk=kk: mis(Xs, n, kk))) for kk in MIS[gname]]
    for kind, label, sims in accounts:
        meas_L = run(X, sims)[0]; meas = (meas_L.argmax(1) == y).mean()
        dS = np.stack([sum((Gp[pos][:, :, n] * (sims[n] - H[:, :, n])).sum(1) for n in ns) for pos in range(10)], 1)
        predL = L + dS; pred = (predL.argmax(1) == y).mean(); agree = (predL.argmax(1) == meas_L.argmax(1)).mean()
        rows.append((meas, pred))
        print(f"{gname:15s} {kind:10s} {label:20s}: measured {meas:.3f}; first order, all ten scores {pred:.3f}; predicted answer = actual answer on {agree:.3f}", flush=True)
R = np.array(rows); print(f"rank correlation {spearmanr(R[:, 0], R[:, 1])[0]:.2f}; median |miss| {np.median(np.abs(R[:, 0] - R[:, 1])):.3f}")
