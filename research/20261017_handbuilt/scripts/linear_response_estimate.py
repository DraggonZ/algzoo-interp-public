"""Linear-response estimate of the accuracy of noisy copies of ARC's model (ARC's cheap MSE protocol).
Advice (computed once from the unperturbed model): for each placement of the top three and each setting,
the exact flip point v* (bisection) and the gradient of the tie margin with respect to all 432 weights and to
the varied value, at v*. Per noisy copy: predicted flip v' = v* - (dM/dW . dW)/(dM/dv); then the exact-band
formula (as in bias_formula_v3.py). Cost per copy: about (configs x 720 x 432) multiply-adds.
Compared with: the true error (4M sampled lists), the exact-band formula run on the copy itself, and sampling."""
import itertools
import json
import sys

import numpy as np
import torch as th

sys.argv = ["x"]
exec(open("scripts/bias_formula_v3.py").read().split("models = {")[0])  # pl, N, r, GRID, WTS, F, F23, F12, bisect, ratio
from algzoo.core import accuracy, example  # noqa

dev = "cuda"
base = example()
G = 3
GVALS = list(np.percentile(ratio, (np.arange(G) + 0.5) / G * 100))


def configs():
    out = []
    for g in GVALS:
        x = np.full((N, 10), -0.46); x[r, pl[:, 0]] = g; x[r, pl[:, 1]] = 1.0
        out.append(("23", x, 2, 0.3, min(1.7, g - 1e-3), 1, 2, 1.0, 1.0, 1 / G))
    x = np.full((N, 10), -0.46); x[r, pl[:, 0]] = 1.6; x[r, pl[:, 2]] = 0.5
    out.append(("12", x, 1, 0.6, 2.6, 1, 0, 1.6, 1.6, 1.0))
    return out


def margin_fn(params, xrow, e):
    wi, wh, wo = params
    h = th.zeros(16, dtype=xrow.dtype, device=dev)
    for t in range(10):
        h = th.relu(wh @ h + wi * xrow[t])
    return (wo @ h) @ e


def advice(P):
    params = tuple(th.tensor(p, device=dev) for p in P)
    adv = []
    for kind, x, var, lo, hi, a, b, ref, scale, w in configs():
        vstar = bisect(P, x, var, lo, hi, a, b, steps=30)
        xx = x.copy(); xx[r, pl[:, var]] = vstar
        X = th.tensor(xx, device=dev)
        E = th.zeros(N, 10, dtype=X.dtype, device=dev)
        E[th.arange(N), th.tensor(pl[:, a], device=dev)] += 1; E[th.arange(N), th.tensor(pl[:, b], device=dev)] -= 1
        gW = th.func.vmap(th.func.grad(margin_fn, argnums=0), in_dims=(None, 0, 0))(params, X, E)
        gx = th.func.vmap(th.func.grad(margin_fn, argnums=1), in_dims=(None, 0, 0))(params, X, E)
        gv = gx[th.arange(N, device=dev), th.tensor(pl[:, var], device=dev)]
        gflat = th.cat([g.reshape(N, -1) for g in gW], 1)
        adv.append(dict(kind=kind, vstar=th.tensor(vstar, device=dev), gW=gflat, gv=gv, lo=lo, hi=hi, ref=ref, scale=scale, w=w))
    return adv


def estimate_from_advice(adv, dW):
    e = 11 / 1024 * 8 / 9
    for c in adv:
        v = c["vstar"] - (c["gW"] @ dW) / c["gv"]
        v = th.clamp(th.nan_to_num(v, nan=1.0), c["lo"], c["hi"]).cpu().numpy()
        bb = (v - c["ref"]) / c["scale"]
        e += c["w"] * np.mean(F(F23 if c["kind"] == "23" else F12, -bb))
    return e


def estimate_exact(P):
    e = 11 / 1024 * 8 / 9
    for kind, x, var, lo, hi, a, b, ref, scale, w in configs():
        bb = (bisect(P, x, var, lo, hi, a, b, steps=24) - ref) / scale
        e += w * np.mean(F(F23 if kind == "23" else F12, -bb))
    return e


adv = advice(base)
flat0 = th.cat([th.tensor(p, device=dev).reshape(-1) for p in base])
rng = np.random.default_rng(2024)
cost_runs = len(adv) * N * 432 / (10 * (16 * 16 + 16) + 160)
res = {}
for sd, n in ((0.005, 24), (0.01, 16), (0.02, 16)):
    rows = []
    for i in range(n):
        P = tuple(w * (1 + sd * rng.standard_normal(w.shape)) for w in base)
        flat = th.cat([th.tensor(p, device=dev).reshape(-1) for p in P])
        lin = estimate_from_advice(adv, flat - flat0)
        exact = estimate_exact(P)
        err = 1 - accuracy(P, n_samples=4 * 2**20, dtype=th.float32, seed=7)[0]
        rows.append((err, lin, exact))
    a = np.array(rows); res[sd] = a.tolist()
    e, lin, ex = a.T
    samp = np.sqrt(np.mean(e * (1 - e) / cost_runs))
    print(f"noise {sd:.1%}: error {100*e.mean():.1f} pts (spread {100*e.std():.2f}); linear-response miss {100*np.sqrt(np.mean((lin-e)**2)):.2f} pts; "
          f"exact-formula miss {100*np.sqrt(np.mean((ex-e)**2)):.2f}; sampling at the same cost ({cost_runs:.0f} runs) {100*samp:.2f}")
json.dump(res, open("linear_response.json", "w"))
