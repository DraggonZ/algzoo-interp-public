"""Sampling-free error estimate from close-call biases, corrected: one-sided bias (a mistake band exists
only when the model favours the smaller of the two), averaged over the distribution of the largest's
value at second-third near-ties (a fact about random numbers). No constant is fitted to accuracy.
error = (at most one positive) x 8/9 + c23 * E[max(0, -b23)] + c12 * E[max(0, -b12)]."""
import itertools
import json
from pathlib import Path

import numpy as np
import torch as th

from algzoo.core import example, forward, load, zoo

pl = np.array(list(itertools.permutations(range(10), 3))); N = len(pl); r = np.arange(N)
GRID = [1.05, 1.15, 1.3, 1.6, 2.0, 3.0, 5.0]


def bisect(P, base, var, lo, hi, a, b, steps=22):
    lo, hi = np.full(N, lo), np.full(N, hi)
    for _ in range(steps):
        mid = (lo + hi) / 2; x = base.copy(); x[r, pl[:, var]] = mid
        s = forward(P, th.tensor(x, device="cuda")).cpu().numpy()
        f = s[r, pl[:, b]] > s[r, pl[:, a]]
        hi = np.where(f, mid, hi); lo = np.where(f, lo, mid)
    return (lo + hi) / 2


def one_sided(P, wts):
    out23 = []
    for g in GRID:
        base = np.full((N, 10), -0.46); base[r, pl[:, 0]] = g; base[r, pl[:, 1]] = 1.0
        b = bisect(P, base, 2, 0.8, 1.2, 1, 2) - 1.0        # third's slot overtakes the second's at 1 + b
        out23.append(np.mean(np.maximum(0, -b)))
    base = np.full((N, 10), -0.46); base[r, pl[:, 0]] = 1.6; base[r, pl[:, 2]] = 0.5
    b12 = (bisect(P, base, 1, 1.3, 1.9, 1, 0) - 1.6) / 1.6   # largest's slot overtakes the second's
    return float(np.dot(wts, out23)), float(np.mean(np.maximum(0, -b12)))


rng = np.random.default_rng(1)
x = np.sort(rng.standard_normal((5_000_000, 10)), 1)
m1, m2, m3 = x[:, -1], x[:, -2], x[:, -3]
ratio = (m1 / m2)[(m2 > 0) & (m2 - m3 < 0.01 * m2)]
edges = np.r_[1.0, [(a + b) / 2 for a, b in zip(GRID[:-1], GRID[1:])], np.inf]
WTS = np.array([np.mean((ratio >= a) & (ratio < b)) for a, b in zip(edges[:-1], edges[1:])])
C23, C12, ONE = 1.926, 1.719, 11 / 1024 * 8 / 9

import sys
FRESH = len(sys.argv) > 1 and sys.argv[1] == "fresh"
models = {} if FRESH else {"ARC": ("trained", example())}
for f in (sorted(Path("/data/output/research/algzoo/20261016_run3/models").glob("2nd_argmax_16_10_*_2*.pth")) if FRESH else []):
    models["fresh" + f.stem.split("_")[-1]] = ("fresh", load(str(f)))
for s in ([] if FRESH else range(5)):
    models[f"zoo{s}"] = ("trained", zoo(16, 10, s))
for f in ([] if FRESH else sorted(Path("/data/output/research/algzoo/20261016_run3/models").glob("2nd_argmax_16_10_*_1*.pth"))):
    models["new" + f.stem.split("_")[-1]] = ("trained", load(str(f)))
rng = np.random.default_rng(123)
for sd, n in (() if FRESH else ((0.005, 32), (0.01, 16))):
    for i in range(n):
        models[f"noise{sd}_{i}"] = (f"noise {sd:.1%}", tuple(w * (1 + sd * rng.standard_normal(w.shape)) for w in example()))
from algzoo.core import accuracy
acc = json.load(open("bias_width_rows.json")) if not FRESH else {k: {"acc": accuracy(P, n_samples=2**24, dtype=th.float32, seed=1)[0]} for k, (_, P) in models.items()}
res = {}
for name, (kind, P) in models.items():
    a, b = one_sided(P, WTS)
    res[name] = dict(kind=kind, err=1 - acc[name]["acc"], pred=ONE + C23 * a + C12 * b, a=a, b=b)
json.dump(res, open("bias_formula_v2_fresh.json" if FRESH else "bias_formula_v2.json", "w"), indent=1)
for kind in (("fresh",) if FRESH else ("trained", "noise 0.5%", "noise 1.0%")):
    ks = [k for k in res if res[k]["kind"] == kind]
    e = np.array([res[k]["err"] for k in ks]); p = np.array([res[k]["pred"] for k in ks])
    print(f"{kind:11s}: error {100*e.mean():.2f} pts on average, predicted {100*p.mean():.2f}; root-mean-square miss {100*np.sqrt(np.mean((p-e)**2)):.2f} pts "
          f"(a constant guess misses by {100*e.std():.2f}); corr {np.corrcoef(p, e)[0,1]:+.2f}")
for k in [k for k in res if res[k]["kind"] in ("trained", "fresh")]:
    print(f"   {k:7s} error {100*res[k]['err']:5.2f}  predicted {100*res[k]['pred']:5.2f}")
