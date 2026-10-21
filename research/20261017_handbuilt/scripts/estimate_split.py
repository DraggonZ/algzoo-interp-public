"""Split each network's estimate error by kind of mistake.
Kinds: A at most one positive number; B answer is the third-largest; C answer is the largest; D anything else.
Estimate terms: A = 11/1024 x 8/9, B = e23, C = e12, D = 0. Measured on 4 million random lists per network.
Estimate code copied from bias_formula_v3.py (which runs on import). Original docstring follows.
Exact-band version of the bias-size estimate.
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
    return 11 / 1024 * 8 / 9, e23, np.mean(F(F12, -b12))




def measured(P, n=4_000_000, seed=7):
    g = th.Generator(device="cuda").manual_seed(seed); out = np.zeros(4)
    for _ in range(n // 1_000_000):
        xx = th.randn(1_000_000, 10, device="cuda", generator=g, dtype=th.float64)
        ans = forward(P, xx).argmax(-1); o = xx.argsort(-1, descending=True)
        wrong = ans != o[:, 1]; few = (xx > 0).sum(-1) <= 1
        A = wrong & few; B = wrong & ~few & (ans == o[:, 2]); C = wrong & ~few & (ans == o[:, 0]); Dd = wrong & ~few & ~(ans == o[:, 2]) & ~(ans == o[:, 0])
        out += np.array([t.double().mean().item() for t in (A, B, C, Dd)])
    return out / (n // 1_000_000)


models = {"ARC": example(), **{f"zoo{s}": zoo(16, 10, s) for s in range(5)}}
for f in sorted(Path("/data/output/research/algzoo/20261016_run3/models").glob("2nd_argmax_16_10_*.pth")):
    models["s" + f.stem.split("_")[-1]] = load(str(f))
rows = {}
print("network | measured A B C D (pts) | estimate A B C (pts) | estimate minus measured: A B C D")
for k, P in models.items():
    m = 100 * measured(P); e = 100 * np.array([*estimate(P), 0.0])
    rows[k] = {"measured": m.tolist(), "estimate": e.tolist()}
    print(f"{k:6s} | {m[0]:.2f} {m[1]:.2f} {m[2]:.2f} {m[3]:.2f} | {e[0]:.2f} {e[1]:.2f} {e[2]:.2f} | {e[0]-m[0]:+.2f} {e[1]-m[1]:+.2f} {e[2]-m[2]:+.2f} {-m[3]:+.2f}")
json.dump(rows, open("estimate_split.json", "w"), indent=1)
for lab, ks in (("22 development", [k for k in rows if not k.startswith("s2")]), ("16 later", [k for k in rows if k.startswith("s2")])):
    d = np.array([np.array(rows[k]["estimate"]) - np.array(rows[k]["measured"]) for k in ks])
    print(f"{lab}: mean estimate minus measured per kind A {d[:,0].mean():+.2f}, B {d[:,1].mean():+.2f}, C {d[:,2].mean():+.2f}, D {d[:,3].mean():+.2f}; "
          f"root-mean-square per kind {np.sqrt((d**2).mean(0)).round(2).tolist()}; total rms {np.sqrt((d.sum(1)**2).mean()):.2f}")
