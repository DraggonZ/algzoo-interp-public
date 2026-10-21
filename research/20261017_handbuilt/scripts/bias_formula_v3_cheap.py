"""Cheaper variants of the exact-band estimate: Newton steps (forward + backward, about 3 runs each) instead of
bisection, fewer values of the largest (G), and fixed evenly spread subsets of the 720 placements (K).
Scored on the 22 trained models (truth from bias_formula_v3.json); cost in network runs."""
import json
import sys

import numpy as np
import torch as th

sys.argv = ["x"]
src = open("scripts/bias_formula_v3.py").read().split("models = {")[0]
exec(src)
from algzoo.core import example, load, zoo  # noqa
from pathlib import Path


def newton(P, plk, which, g, steps):
    n = len(plk); rr = np.arange(n)
    if which == "23":
        v = np.full(n, 1.0 - 1e-6); lo, hi, var, a, b, scale, ref = 0.3, min(1.7, g - 1e-3), 2, 1, 2, 1.0, 1.0
    else:
        v = np.full(n, g * (1 - 1e-6)); lo, hi, var, a, b, scale, ref = 0.6, 2.6, 1, 1, 0, g, g
    for _ in range(steps):
        xx = np.full((n, 10), -0.46); xx[rr, plk[:, 0]] = g
        if which == "23":
            xx[rr, plk[:, 1]] = 1.0; xx[rr, plk[:, 2]] = v
        else:
            xx[rr, plk[:, 1]] = v; xx[rr, plk[:, 2]] = 0.5
        X = th.tensor(xx, device="cuda", requires_grad=True)
        s = forward(P, X); M = s[rr, plk[:, a]] - s[rr, plk[:, b]]
        gr = th.autograd.grad(M.sum(), X)[0][rr, plk[:, var]]
        step = np.nan_to_num((-(M / gr)).detach().cpu().numpy(), nan=0.0, posinf=1, neginf=-1)
        v = np.clip(v + step, lo, hi)
    return (v - ref) / scale


def cheap(P, K, Gpts, steps):
    idx = np.linspace(0, len(pl) - 1, K).round().astype(int); plk = pl[idx]
    if Gpts == len(GRID):
        grid, w = GRID, WTS
    else:
        grid = list(np.percentile(ratio, (np.arange(Gpts) + 0.5) / Gpts * 100)); w = np.full(Gpts, 1 / Gpts)
    e23 = sum(wi * np.mean(F(F23, -newton(P, plk, "23", g, steps))) for g, wi in zip(grid, w))
    e12 = np.mean(F(F12, -newton(P, plk, "12", 1.6, steps)))
    return 11 / 1024 * 8 / 9 + e23 + e12, 3 * steps * K * (len(grid) + 1)


rows = json.load(open("bias_formula_v3.json"))
models = {"ARC": example(), **{f"zoo{s}": zoo(16, 10, s) for s in range(5)}}
for f in sorted(Path("/data/output/research/algzoo/20261016_run3/models").glob("2nd_argmax_16_10_*_1*.pth")):
    models["s" + f.stem.split("_")[-1]] = load(str(f))
ks = list(models); e = np.array([rows[k][0] for k in ks])
for K, Gp, st in ((720, 7, 3), (720, 3, 2), (720, 3, 1), (240, 3, 2), (72, 3, 2), (72, 1, 2), (24, 1, 2)):
    out = [cheap(models[k], K, Gp, st) for k in ks]
    p = np.array([o[0] for o in out]); cost = out[0][1]
    miss = 100 * np.sqrt(np.mean((p - e) ** 2)); samp = 100 * np.sqrt(np.mean(e * (1 - e) / cost))
    print(f"K={K:3d} G={Gp} steps={st}: cost {cost:6d} runs; miss {miss:.2f} pts; sampling at that cost {samp:.2f}; squared-error ratio {(miss/samp)**2:.1f}")
