"""Across the 22 trained networks: one step after a record, do some hiding neurons compare the arriving number with
the previous largest (now the second-largest) rather than with the new largest? Hiding neurons by the 85% rule.
For each, fit its pre-activation one step after a record on (previous largest p, new largest m1, arriving number x)
and report the weights and variance matched. A neuron 'compares with the second-largest' when its weight on p is
larger than its weight on m1."""
from pathlib import Path
import numpy as np
from algzoo.core import example, zoo, load
models = {"ARC": example()}
for s in range(5): models[f"zoo{s}"] = zoo(16, 10, s)
for f in sorted(Path("/data/output/research/algzoo/20261016_run3/models").glob("2nd_argmax_16_10_*_1*.pth")):
    models["new" + f.stem.split("_")[-1]] = load(str(f))
rng = np.random.default_rng(14); X = rng.standard_normal((400000, 10)); N = len(X)
pm = np.maximum.accumulate(np.c_[np.full(N, -np.inf), X], 1)[:, :-1]; rec = X > pm
ar = np.arange(10)[None, :]; recm = rec & (ar >= 1); oth = ~rec & (ar >= 2)
n_with = 0; n_nets = 0
for name, w in models.items():
    wi, wh, wo = (np.asarray(p, float) for p in w); wi = wi.reshape(-1)
    h = np.zeros((N, 16)); HS = []; PRE = []
    for t in range(10):
        pre = h @ wh.T + X[:, t:t + 1] * wi; h = np.maximum(pre, 0); HS.append(h); PRE.append(pre)
    HS = np.stack(HS, 1); PRE = np.stack(PRE, 1)
    hid = [n for n in range(16) if (HS[:, :, n][recm] <= 1e-9).mean() >= 0.85 and (HS[:, :, n][oth] > 1e-9).mean() >= 0.5]
    if not hid: print(f"{name:7s} no hiding neurons"); continue
    rows = []
    for r in range(2, 7):
        p = np.maximum(X[:, :r].max(1), 0); ok = (X[:, r] > p + 0.05) & (p > 0.2)
        rows.append(np.c_[p[ok], X[ok, r], X[ok, r + 1], PRE[ok, r + 1][:, hid]])
    R = np.concatenate(rows); A = R[:, :3]; out = []; comp = 0
    for i, n in enumerate(hid):
        b = R[:, 3 + i]; c = np.linalg.lstsq(A, b, rcond=None)[0]; e = b - A @ c
        tag = "second" if c[0] > c[1] else "largest"; comp += tag == "second"
        out.append(f"{n}: {c[0]:5.1f} p {c[1]:+5.1f} m1 {c[2]:+5.1f} x ({1 - e.var() / b.var():.2f}, {tag})")
    n_nets += 1; n_with += comp > 0
    print(f"{name:7s} " + " | ".join(out))
print(f"networks with a hiding neuron that compares with the second-largest one step after a record: {n_with} of {n_nets}")
