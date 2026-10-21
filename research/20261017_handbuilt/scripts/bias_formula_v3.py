"""Exact-band version of the bias-size estimate.
Per placement and per value of the largest: one-sided bias from bisection over a wide range (no 0.2 cap).
Mistake probability for that placement = F(bias), where F(d) = P(relative gap in (0, d) and second > 0)
for random normal lists, tabulated (a fact about random numbers), instead of density x width.
error = 11/1024 x 8/9 + mean_placements F23(max(0, -b23)) + mean_placements F12(max(0, -b12))."""
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import torch as th

from algzoo.core import accuracy, example, forward, load, zoo

pl = np.array(list(itertools.permutations(range(10), 3))); N = len(pl); r = np.arange(N)
GRID = [1.05, 1.15, 1.3, 1.6, 2.0, 3.0, 5.0]
rng = np.random.default_rng(1)
x = np.sort(rng.standard_normal((5_000_000, 10)), 1)
m1, m2, m3 = x[:, -1], x[:, -2], x[:, -3]
pos = m2 > 0
rel23, rel12 = np.where(pos, (m2 - m3) / np.abs(m2), np.inf), np.where(pos, (m1 - m2) / m1, np.inf)
ratio = (m1 / m2)[pos & (rel23 < 0.01)]
edges = np.r_[1.0, [(a + b) / 2 for a, b in zip(GRID[:-1], GRID[1:])], np.inf]
WTS = np.array([np.mean((ratio >= a) & (ratio < b)) for a, b in zip(edges[:-1], edges[1:])])
D = np.linspace(0, 1, 2001)
F23 = np.array([np.mean(rel23 < d) for d in D]); F12 = np.array([np.mean(rel12 < d) for d in D])
F = lambda tab, b: np.interp(np.clip(b, 0, 1), D, tab)


def bisect(P, base, var, lo, hi, a, b, steps=24):
    lo, hi = np.full(N, lo), np.full(N, hi)
    for _ in range(steps):
        mid = (lo + hi) / 2; xx = base.copy(); xx[r, pl[:, var]] = mid
        s = forward(P, th.tensor(xx, device="cuda")).cpu().numpy()
        f = s[r, pl[:, b]] > s[r, pl[:, a]]
        hi = np.where(f, mid, hi); lo = np.where(f, lo, mid)
    return (lo + hi) / 2


def estimate(P):
    e23 = 0.0
    for g, w in zip(GRID, WTS):
        base = np.full((N, 10), -0.46); base[r, pl[:, 0]] = g; base[r, pl[:, 1]] = 1.0
        b = bisect(P, base, 2, 0.3, min(1.7, g - 1e-3), 1, 2) - 1.0
        e23 += w * np.mean(F(F23, -b))
    base = np.full((N, 10), -0.46); base[r, pl[:, 0]] = 1.6; base[r, pl[:, 2]] = 0.5
    b12 = (bisect(P, base, 1, 0.6, 2.6, 1, 0) - 1.6) / 1.6
    return 11 / 1024 * 8 / 9 + e23 + np.mean(F(F12, -b12)), e23


models = {"ARC": example(), **{f"zoo{s}": zoo(16, 10, s) for s in range(5)}}
for f in sorted(Path("/data/output/research/algzoo/20261016_run3/models").glob("2nd_argmax_16_10_*.pth")):
    models["s" + f.stem.split("_")[-1]] = load(str(f))
truth = {}
for k, v in json.load(open("bias_width_rows.json")).items():
    if v["kind"] == "trained":
        truth["s" + k[3:] if k.startswith("new") else k] = 1 - v["acc"]
for k, v in json.load(open("bias_formula_v2_fresh.json")).items():
    truth["s" + k[5:]] = v["err"]
rows = {}
for k, P in models.items():
    pred, e23 = estimate(P)
    rows[k] = (truth[k], pred)
json.dump(rows, open("bias_formula_v3.json", "w"), indent=1)
for lab, ks in (("22 trained", [k for k in rows if not k.startswith("s2")]), ("16 fresh (already seen)", [k for k in rows if k.startswith("s2")])):
    e = np.array([rows[k][0] for k in ks]); p = np.array([rows[k][1] for k in ks])
    no208 = [i for i, k in enumerate(ks) if k != "s208"]
    print(f"{lab}: miss {100*np.sqrt(np.mean((p-e)**2)):.2f} pts (without seed 208: {100*np.sqrt(np.mean((p-e)[no208]**2)):.2f}); mean signed {100*np.mean(p-e):+.2f}; corr {np.corrcoef(p,e)[0,1]:+.2f}; constant guess {100*e.std():.2f}")
for k in rows:
    print(f"  {k:6s} error {100*rows[k][0]:5.2f} predicted {100*rows[k][1]:5.2f}")
