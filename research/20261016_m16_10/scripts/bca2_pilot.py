"""Pilot 2 of the biased-comparator algorithm: comparisons plus pointer-operation fidelities, fitted per model.

Adds to bca_pilot.py: on each step the believed event (ordinary, new 2nd, new max) is carried out correctly
with a probability measured on clear-cut steps of the network (fidelity.py), by age. A failed operation loses
the pointer; a lost pointer stays lost until a later new 2nd or new max is carried out correctly.
Still data-fitted (an upper bound on the abstraction), not mechanistic.
"""
import json
import time

import numpy as np
import torch as th

from bca_pilot import params_from_table
from bench import load_bench
from core import example
from fidelity import fidelity
from thresholds import threshold_table

dev = "cuda"


def fidelity_tables(stats):
    """Arrays F[event][age1, age2] with fallback to the row/column marginals, then to the overall rate."""
    out = {}
    for k in ("ordinary", "new 2nd", "new max"):
        C = np.zeros((9, 9)); Nn = np.zeros((9, 9))
        for (kk, a1, a2), (c, n) in stats.items():
            if kk == k and a1 < 9 and a2 < 9:
                C[a1, a2] += c; Nn[a1, a2] += n
        overall = C.sum() / max(Nn.sum(), 1)
        r1 = (C.sum(1) + 50 * overall) / (Nn.sum(1) + 50)
        r2 = (C.sum(0) + 50 * overall) / (Nn.sum(0) + 50)
        F = np.where(Nn >= 200, C / np.maximum(Nn, 1), np.nan)
        if k == "new max":                   # depends mainly on the age of the max (clock)
            fb = np.repeat(r1[:, None], 9, 1)
        else:                                # shifts and writes: age of the 2nd matters most
            fb = np.repeat(r2[None, :], 9, 0)
        out[k] = np.where(np.isfinite(F), F, fb)
    return out


@th.no_grad()
def simulate(th1, w1, th2, w2, fid, N=2 * 10**6, seed=0, t_fid=5):
    g = th.Generator(dev).manual_seed(seed)
    x = th.randn(N, 10, device=dev, dtype=th.float64, generator=g)
    u = th.rand(N, 10, 3, device=dev, dtype=th.float64, generator=g)
    T = lambda a: th.as_tensor(np.asarray(a), device=dev, dtype=th.float64)
    th1, w1, th2, w2 = T(th1), T(w1), T(th2), T(w2)
    Ford, F2, Fmax = T(fid["ordinary"]), T(fid["new 2nd"]), T(fid["new max"])
    M1 = th.zeros(N, device=dev, dtype=th.float64); P1 = th.full((N,), -1, device=dev)
    M2 = th.zeros(N, device=dev, dtype=th.float64); P2 = th.full((N,), -1, device=dev)
    ok = th.ones(N, dtype=th.bool, device=dev)
    for t in range(10):
        xt = x[:, t]
        age1 = (t - 1 - P1).clamp(0, 8); age2 = (t - 1 - P2).clamp(0, 8)
        r1 = th.where(M1 > 0, xt / M1.clamp(min=1e-12), th.where(xt > 0, th.full_like(xt, 9.0), th.full_like(xt, -9.0)))
        new_max = u[:, t, 0] < th.sigmoid((r1 - th1[age1]) / w1[age1])
        r2 = th.where(M2 > 0, xt / M2.clamp(min=1e-12), th.where(xt > 0, th.full_like(xt, 9.0), th.full_like(xt, -9.0)))
        new_2 = (~new_max) & (u[:, t, 1] < th.sigmoid((r2 - th2[age2]) / w2[age2]))
        ordinary = ~new_max & ~new_2
        if t >= t_fid:
            pf = th.where(new_max, Fmax[age1, age2], th.where(new_2, F2[age1, age2], Ford[age1, age2]))
            success = u[:, t, 2] < pf
        else:
            success = th.ones_like(ok)
        ok = th.where(ordinary, ok & success, success)     # a correct rewrite restores the pointer
        M2n = th.where(new_max, M1, th.where(new_2, xt.clamp(min=0), M2))
        P2n = th.where(new_max, P1, th.where(new_2, th.full_like(P2, t), P2))
        M1 = th.where(new_max, xt.clamp(min=0), M1); P1 = th.where(new_max, th.full_like(P1, t), P1)
        M2, P2 = M2n, P2n
    tgt = x.argsort(-1)[:, -2]
    return (ok & (P2 == tgt)).double().mean().item()


if __name__ == "__main__":
    names, acc, models = load_bench(("mult",))
    names = ["example"] + list(names); acc = np.concatenate([[0.95314], acc]); models = [example()] + models
    rows = []
    t0 = time.time()
    for i, (n, a, m) in enumerate(zip(names, acc, models)):
        pars = params_from_table(threshold_table(m, N=2 * 10**6, seed=100 + i))
        fid = fidelity_tables(fidelity(m, N=2 * 10**6, seed=300 + i))
        b = simulate(*pars, fid, seed=200 + i)
        rows.append((str(n), float(a), float(b)))
        print(f"{n:16s} true {a:.4f}  BCA2 {b:.4f}  ({time.time() - t0:.0f}s)", flush=True)
    json.dump(rows, open("/data/output/research/algzoo/20261016_m16_10/reports/bca2_pilot.json", "w"))
    true = np.array([r[1] for r in rows[1:]]); est = np.array([r[2] for r in rows[1:]]); e = est - true
    print(f"96 copies: RMSE {np.sqrt(np.mean(e**2)):.4f}, bias {e.mean():+.4f}, corr {np.corrcoef(est, true)[0, 1]:.3f}")
    for s in ["0.005", "0.01", "0.02"]:
        sel = np.array([r[0].startswith("mult" + s) for r in rows[1:]])
        print(f"  sigma {s}: RMSE {np.sqrt(np.mean(e[sel]**2)):.4f}, bias {e[sel].mean():+.4f}, corr {np.corrcoef(est[sel], true[sel])[0, 1]:.3f}")
