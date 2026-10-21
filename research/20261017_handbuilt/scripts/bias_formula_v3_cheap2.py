"""Cheaper bisection variants of the exact-band estimate (22 trained models): fewer bisection steps,
fewer values of the largest (G, equal-mass bins of the near-tie distribution), fixed placement subsets (K)."""
import json
import sys
from pathlib import Path

import numpy as np
import torch as th

sys.argv = ["x"]
exec(open("scripts/bias_formula_v3.py").read().split("models = {")[0])
from algzoo.core import example, load, zoo  # noqa


def bis(P, plk, base, var, lo, hi, a, b, steps):
    n = len(plk); rr = np.arange(n); lo, hi = np.full(n, lo), np.full(n, hi)
    for _ in range(steps):
        mid = (lo + hi) / 2; xx = base.copy(); xx[rr, plk[:, var]] = mid
        s = forward(P, th.tensor(xx, device="cuda")).cpu().numpy()
        f = s[rr, plk[:, b]] > s[rr, plk[:, a]]
        hi = np.where(f, mid, hi); lo = np.where(f, lo, mid)
    return (lo + hi) / 2


def cheap(P, K, G, steps):
    idx = np.linspace(0, len(pl) - 1, K).round().astype(int); plk = pl[idx]; n = K; rr = np.arange(n)
    grid = list(np.percentile(ratio, (np.arange(G) + 0.5) / G * 100))
    e23 = 0
    for g in grid:
        base = np.full((n, 10), -0.46); base[rr, plk[:, 0]] = g; base[rr, plk[:, 1]] = 1.0
        e23 += np.mean(F(F23, -(bis(P, plk, base, 2, 0.3, min(1.7, g - 1e-3), 1, 2, steps) - 1))) / G
    base = np.full((n, 10), -0.46); base[rr, plk[:, 0]] = 1.6; base[rr, plk[:, 2]] = 0.5
    e12 = np.mean(F(F12, -(bis(P, plk, base, 1, 0.6, 2.6, 1, 0, steps) - 1.6) / 1.6))
    return 11 / 1024 * 8 / 9 + e23 + e12, steps * K * (G + 1)


rows = json.load(open("bias_formula_v3.json"))
models = {"ARC": example(), **{f"zoo{s}": zoo(16, 10, s) for s in range(5)}}
for f in sorted(Path("/data/output/research/algzoo/20261016_run3/models").glob("2nd_argmax_16_10_*_1*.pth")):
    models["s" + f.stem.split("_")[-1]] = load(str(f))
ks = list(models); e = np.array([rows[k][0] for k in ks])
for K, G, st in ((720, 7, 12), (720, 3, 10), (720, 2, 8), (240, 3, 8), (240, 2, 7), (120, 2, 7), (72, 2, 7), (72, 1, 7), (36, 1, 6), (12, 1, 6)):
    out = [cheap(models[k], K, G, st) for k in ks]
    p = np.array([o[0] for o in out]); cost = out[0][1]
    miss = 100 * np.sqrt(np.mean((p - e) ** 2)); samp = 100 * np.sqrt(np.mean(e * (1 - e) / cost))
    print(f"K={K:3d} G={G} steps={st:2d}: cost {cost:6d} runs; miss {miss:.2f} pts; sampling at that cost {samp:.2f}; squared-error ratio {(miss/samp)**2:.1f}")
