"""Pilot of the biased-comparator algorithm (BCA), with thresholds fitted from each model's behaviour.

BCA: the ideal 2nd-argmax algorithm, but each comparison of a new input x with a stored value m (max or 2nd,
with the network's 0 floor) succeeds with probability sigmoid((x/m - theta(age)) / width(age)), age being steps
since the stored value arrived. The algorithm tracks its own beliefs (stored values, positions, ages), so a
missed comparison changes what later inputs are compared with. Output: the believed 2nd max position.

This is a data-fitted upper bound on what the abstraction can do. It is not mechanistic: thresholds come from
sampling the network. Kill rule (review 1): RMSE above 0.04 or correlation below 0.9 across the 96 copies.
"""
import json
import time

import numpy as np
import torch as th

from bench import load_bench
from core import example
from thresholds import threshold_table

dev = "cuda"


def params_from_table(tab):
    th1, w1, th2, w2 = (np.ones(9), np.full(9, 0.01), np.ones(9), np.full(9, 0.03))
    for age in range(9):
        c, w, n = tab[("max", age)]
        if np.isfinite(c) and np.isfinite(w):
            th1[age], w1[age] = c, max(w, 1e-4)
        elif age > 0:
            th1[age], w1[age] = th1[age - 1], w1[age - 1]
        c, w, n = tab[("2nd", age)]
        if np.isfinite(c) and np.isfinite(w):
            th2[age], w2[age] = c, max(w, 1e-4)
        elif age > 0:
            th2[age], w2[age] = th2[age - 1], w2[age - 1]
    return th1, w1, th2, w2


@th.no_grad()
def simulate(th1, w1, th2, w2, N=2 * 10**6, seed=0):
    g = th.Generator(dev).manual_seed(seed)
    x = th.randn(N, 10, device=dev, dtype=th.float64, generator=g)
    u = th.rand(N, 10, 2, device=dev, dtype=th.float64, generator=g)
    T = lambda a: th.as_tensor(a, device=dev, dtype=th.float64)
    th1, w1, th2, w2 = T(th1), T(w1), T(th2), T(w2)
    # beliefs: stored max value (0 floor), its position; stored 2nd value (0 floor), its position (-1 = none)
    M1 = th.zeros(N, device=dev, dtype=th.float64); P1 = th.full((N,), -1, device=dev)
    M2 = th.zeros(N, device=dev, dtype=th.float64); P2 = th.full((N,), -1, device=dev)
    for t in range(10):
        xt = x[:, t]
        age1 = (t - 1 - P1).clamp(0, 8)
        age2 = (t - 1 - P2).clamp(0, 8)
        # comparison with the stored max; with an empty (zero) store, compare with 0 exactly
        r1 = th.where(M1 > 0, xt / M1.clamp(min=1e-12), th.where(xt > 0, th.full_like(xt, 9.0), th.full_like(xt, -9.0)))
        p_new_max = th.sigmoid((r1 - th1[age1]) / w1[age1])
        new_max = u[:, t, 0] < p_new_max
        r2 = th.where(M2 > 0, xt / M2.clamp(min=1e-12), th.where(xt > 0, th.full_like(xt, 9.0), th.full_like(xt, -9.0)))
        p_new_2 = th.sigmoid((r2 - th2[age2]) / w2[age2])
        new_2 = (~new_max) & (u[:, t, 1] < p_new_2)
        # updates
        M2n = th.where(new_max, M1, th.where(new_2, xt.clamp(min=0), M2))
        P2n = th.where(new_max, P1, th.where(new_2, th.full_like(P2, t), P2))
        M1 = th.where(new_max, xt.clamp(min=0), M1)
        P1 = th.where(new_max, th.full_like(P1, t), P1)
        M2, P2 = M2n, P2n
    tgt = x.argsort(-1)[:, -2]
    return (P2 == tgt).double().mean().item()


if __name__ == "__main__":
    names, acc, models = load_bench(("mult",))
    names = ["example"] + list(names)
    acc = np.concatenate([[0.95314], acc])
    models = [example()] + models
    rows = []
    t0 = time.time()
    for i, (n, a, m) in enumerate(zip(names, acc, models)):
        tab = threshold_table(m, N=2 * 10**6, seed=100 + i)
        pars = params_from_table(tab)
        b = simulate(*pars, seed=200 + i)
        rows.append((str(n), float(a), float(b), [p.tolist() for p in pars]))
        print(f"{n:16s} true {a:.4f}  BCA {b:.4f}  ({time.time() - t0:.0f}s)", flush=True)
    json.dump(rows, open("/data/output/research/algzoo/20261016_m16_10/reports/bca_pilot.json", "w"))
    true = np.array([r[1] for r in rows[1:]]); est = np.array([r[2] for r in rows[1:]])
    e = est - true
    print(f"96 copies: RMSE {np.sqrt(np.mean(e**2)):.4f}, bias {e.mean():+.4f}, corr {np.corrcoef(est, true)[0, 1]:.3f}")
    for s in ["0.005", "0.01", "0.02"]:
        sel = np.array([r[0].startswith("mult" + s) for r in rows[1:]])
        print(f"  sigma {s}: RMSE {np.sqrt(np.mean(e[sel]**2)):.4f}, bias {e[sel].mean():+.4f}, corr {np.corrcoef(est[sel], true[sel])[0, 1]:.3f}")
