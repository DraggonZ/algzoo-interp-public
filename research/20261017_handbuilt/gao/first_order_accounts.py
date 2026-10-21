"""Test: do the first-order (attribution-patching) predictions put every account of shuffled_control.py
(ours and the misreadings, one group replaced at a time) and its shuffled-error control on one curve? For each:
measured accuracy against first-order predicted accuracy (g from the true run, error = substituted - true values)."""
import json
import numpy as np
from scipy.stats import spearmanr
import simlib
from simlib import run
from matched_size import X, y, L, H, G, MIS, mis, account_features, fitted

r = np.arange(len(X)); _, P, _ = run(X); k = L.argmax(1); j = np.argsort(L, 1)[:, -2]; lead = L[r, k] - L[r, j]
g = simlib.WO[k] - simlib.WO[j]; Gt = np.zeros_like(H)
for t in range(9, -1, -1):
    Gt[:, t] = g; g = (g * (P[:, t] > 0)) @ simlib.WH
pred = lambda err: np.where(lead + sum((Gt[:, :, n] * e).sum(1) for n, e in err.items()) < 0, j == y, k == y).mean()
rng = np.random.default_rng(1); rows = []
for gname, ns in G.items():
    accounts = [("ours", "account", fitted(gname, account_features))] + [("misreading", kk, fitted(gname, lambda Xs, n, kk=kk: mis(Xs, n, kk))) for kk in MIS[gname]]
    for kind, label, sims in accounts:
        err = {n: sims[n] - H[:, :, n] for n in ns}; perm = rng.permutation(len(X))
        shuf = {n: np.maximum(H[:, :, n] + err[n][perm], 0) for n in ns}; serr = {n: shuf[n] - H[:, :, n] for n in ns}
        for variant, sm, e in (("substituted", sims, err), ("shuffled", shuf, serr)):
            m = (run(X, sm)[0].argmax(1) == y).mean(); p = pred(e)
            rows.append(dict(group=gname, kind=kind, label=label, variant=variant, measured=float(m), predicted=float(p)))
            print(f"{gname:15s} {kind:10s} {label:20s} {variant:11s}: measured {m:.3f}; first-order predicted {p:.3f}", flush=True)
R = [(x["measured"], x["predicted"]) for x in rows]; R = np.array(R)
rho, pv = spearmanr(R[:, 0], R[:, 1])
print(f"all {len(R)} cases: rank correlation {rho:.2f}; median |measured - predicted| {np.median(np.abs(R[:, 0] - R[:, 1])):.3f}")
for kind in ("ours", "misreading"):
    d = [x["measured"] - x["predicted"] for x in rows if x["kind"] == kind and x["variant"] == "substituted"]
    print(f"{kind:10s} substituted: measured minus predicted, median {np.median(d):+.3f}, range {min(d):+.3f} to {max(d):+.3f}")
d = [x["measured"] - x["predicted"] for x in rows if x["variant"] == "shuffled"]
print(f"shuffled controls: measured minus predicted, median {np.median(d):+.3f}, range {min(d):+.3f} to {max(d):+.3f}")
json.dump(rows, open("first_order_accounts.json", "w"), indent=1)
