"""Does the kind of error matter, at exactly the same size? For each account of one neuron group (as matched_size.py),
substitute (a) the account's values, (b) the true values plus the account's own errors shuffled across lists (same
neurons, same steps, so the same size and spread at every neuron; only which list gets which error changes). If (a)
costs much more than (b), the error's structure (kind) matters beyond its size. Trained network, 50,000 lists. Also the
error size at the neurons: RMS of (account - true) over the group's active values, as a share of their typical value."""
import json
import numpy as np
import simlib
from simlib import lists, run
from matched_size import X, y, L, H, Xf, PREf, G, MIS, mis, account_features, fitted

rng = np.random.default_rng(1)
rows = []
for g, ns in G.items():
    accounts = [("ours", "account", fitted(g, account_features))] + [("misreading", k, fitted(g, lambda Xs, n, k=k: mis(Xs, n, k))) for k in MIS[g]]
    for kind, label, sims in accounts:
        err = {n: sims[n] - H[:, :, n] for n in ns}
        perm = rng.permutation(len(X))
        shuf = {n: np.maximum(H[:, :, n] + err[n][perm], 0) for n in ns}
        a_acc = (run(X, sims)[0].argmax(1) == y).mean(); s_acc = (run(X, shuf)[0].argmax(1) == y).mean()
        size = np.sqrt(np.mean([np.mean(err[n][H[:, :, n] > 0] ** 2) / np.median(H[:, :, n][H[:, :, n] > 0]) ** 2 for n in ns]))
        rows.append(dict(group=g, kind=kind, label=label, size=float(size), acc=float(a_acc), shuffled=float(s_acc)))
        print(f"{g:15s} {kind:10s} {label:20s}: error at the neurons {size:.3f} of typical value; accuracy {a_acc:.3f}, with its errors shuffled {s_acc:.3f}", flush=True)
json.dump(rows, open("shuffled_control.json", "w"), indent=1)
