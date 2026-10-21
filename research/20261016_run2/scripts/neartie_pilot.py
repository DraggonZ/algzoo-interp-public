"""Linearized near-tie width (pilot aimed at length 10; specified, with its kill rule, before running).

For each two-spike angle cell of exact_z2.py where the two-spike answer is right, and each low position q, run the
network with the low input set equal to the 2nd max (x_q = m2), take the margin M (target minus best other logit) and
its derivative D in that input, and set width = M / D (zero if M >= 0, the whole range if the margin does not recover
below m2). By homogeneity the band scales with r. The single-input loss for (cell, q) at radius r is the Gaussian mass
of [m2 - w r, m2] given x < m2, and the estimate is the loss-only single-input rule:
    sum over cells of mass(cell) * E_r[ prod_q (1 - loss_q(r)) ]   (cells where the two-spike answer is right).

Kill rule (fixed before running): on the base model, if the linearized width differs from the scanned width by more
than a factor of two on most of the mass-weighted (cell, q) cases, stop and record a closed branch.

Usage: python neartie_pilot.py check     -> kill-rule comparison on the base model
       python neartie_pilot.py score     -> estimator on zoo seeds 0-4 (primary) and the 96 copies per σ
"""
import sys

import numpy as np
import torch as th
from scipy.special import ndtr

from algzoo.exact_z2 import R, WR, angle_mass, correct_intervals, pair_intervals

dev = "cuda"
STEP = 16 * 16 + 16


def cells_of(params):
    """(p1, p2, θ_mid, mass) for every right two-spike cell; also the exact Z_2 and the op count of finding them."""
    ops = [0]
    out = []
    for p1 in range(10):
        for p2 in range(10):
            if p1 == p2:
                continue
            cells = pair_intervals(params, p1, p2, ops)
            for lo, hi in correct_intervals(params, p1, p2, cells, ops):
                out.append((p1, p2, 0.5 * (lo + hi), angle_mass(lo, hi)))
    return out, ops[0]


@th.no_grad()
def widths(params, cells):
    """Linearized width per (cell, q) at r = 1; also the multiply-add count (value + tangent runs from q on)."""
    w_hi, w_hh, w_oh = (th.as_tensor(p, device=dev, dtype=th.float64) for p in params)
    rows, X, Q = [], [], []
    for c, (p1, p2, t, m) in enumerate(cells):
        for q in range(10):
            if q in (p1, p2):
                continue
            x = np.zeros(10)
            x[p1], x[p2], x[q] = np.cos(t), np.sin(t), np.sin(t)
            rows.append((c, q, p2))
            X.append(x)
            Q.append(q)
    X = th.as_tensor(np.array(X), device=dev)
    Q = th.as_tensor(Q, device=dev)
    P2 = th.as_tensor([r[2] for r in rows], device=dev)
    n = len(rows)
    h = th.zeros(n, 16, device=dev, dtype=th.float64)
    dh = th.zeros_like(h)
    for t in range(10):
        z = h @ w_hh.T + X[:, t, None] * w_hi
        dz = dh @ w_hh.T + (Q == t).double()[:, None] * w_hi
        on = (z > 0).double()
        h, dh = z * on, dz * on
    L, dL = h @ w_oh.T, dh @ w_oh.T
    Lo = L.clone()
    Lo[th.arange(n), P2] = -th.inf
    k = Lo.argmax(-1)
    M = L[th.arange(n), P2] - L[th.arange(n), k]
    D = dL[th.arange(n), P2] - dL[th.arange(n), k]
    w = th.where(M >= 0, th.zeros_like(M), th.where(D < 0, M / D, th.full_like(M, np.inf)))
    ops = sum(2 * (10 - q) * STEP + 2 * 16 * 10 for q in Q.tolist()) + len(cells) * 10 * STEP   # tangent + value runs
    return np.array(rows), w.cpu().numpy(), ops


@th.no_grad()
def scanned(params, cells, rows, n_grid=4000, smax=4.0):
    """Scanned width at r = 1: x = m2 - s, s in (0, smax]; width of the failing interval starting just below m2."""
    w_hi, w_hh, w_oh = (th.as_tensor(p, device=dev, dtype=th.float64) for p in params)
    s = th.linspace(smax / n_grid, smax, n_grid, device=dev, dtype=th.float64)
    out = np.zeros(len(rows))
    B = 64
    for i0 in range(0, len(rows), B):
        chunk = rows[i0:i0 + B]
        X = th.zeros(len(chunk), n_grid, 10, device=dev, dtype=th.float64)
        p2s = []
        for j, (c, q, p2) in enumerate(chunk):
            p1_, p2_, t, m = cells[c]
            X[j, :, p1_], X[j, :, p2_] = np.cos(t), np.sin(t)
            X[j, :, q] = np.sin(t) - s
            p2s.append(p2_)
        h = th.zeros(len(chunk), n_grid, 16, device=dev, dtype=th.float64)
        for t in range(10):
            h = th.relu(h @ w_hh.T + X[..., t, None] * w_hi)
        ok = (h @ w_oh.T).argmax(-1) == th.as_tensor(p2s, device=dev)[:, None]     # [chunk, grid]
        okn = ok.cpu().numpy()
        for j in range(len(chunk)):
            if okn[j, 0]:
                out[i0 + j] = 0.0
            elif not okn[j].any():
                out[i0 + j] = np.inf
            else:
                out[i0 + j] = s[np.argmax(okn[j])].item()
    return out


FULL_RANGE = "inf"   # "inf": a margin that does not recover below m2 fails the whole low range (as first run);
#                     "zero": treat it as no band (post-hoc, chosen after seeing the base model)


def estimate(params):
    cells, ops_cells = cells_of(params)
    rows, w, ops_w = widths(params, cells)
    if FULL_RANGE == "zero":
        w = np.where(np.isfinite(w), w, 0.0)
    sth = np.array([np.sin(cells[c][2]) for c in rows[:, 0]])
    # radial density per cell for θ_mid: r e^{-r²/2} Φ(r sinθ)^8, normalised
    tot = 0.0
    surv = {}
    for i, (c, q, p2) in enumerate(rows):
        ms = R * sth[i]
        lo = R * (sth[i] - w[i]) if np.isfinite(w[i]) else np.full_like(R, -np.inf)
        loss = (ndtr(ms) - ndtr(lo)) / np.maximum(ndtr(ms), 1e-300)
        surv[c] = surv.get(c, 1.0) * (1 - loss)
    for c, (p1, p2, t, m) in enumerate(cells):
        dens = R * np.exp(-R**2 / 2) * ndtr(R * np.sin(t)) ** 8 * WR
        s_c = surv.get(c, np.ones_like(R))
        tot += m * (dens * s_c).sum() / dens.sum()
    ops = ops_cells + ops_w + len(rows) * len(R) * 4
    return tot, ops


if __name__ == "__main__":
    from algzoo.bench import load_bench
    from algzoo.core import example, zoo

    if sys.argv[1] == "check":
        params = example()
        cells, _ = cells_of(params)
        rows, w, _ = widths(params, cells)
        ws = scanned(params, cells, rows)
        mass = np.array([cells[c][3] for c in rows[:, 0]])
        zero_l, zero_s = w < 1e-3, ws < 1e-3
        inf_l, inf_s = ~np.isfinite(w) | (w > 3.9), ~np.isfinite(ws) | (ws > 3.9)
        both_fin = ~zero_l & ~zero_s & ~inf_l & ~inf_s
        ratio = np.where(both_fin, w / np.where(ws > 0, ws, 1), np.nan)
        agree = (zero_l & zero_s) | (inf_l & inf_s) | (both_fin & (ratio >= 0.5) & (ratio <= 2))
        share = (mass * agree).sum() / mass.sum()
        print(f"base model: {len(cells)} right two-spike cells, {len(rows)} (cell, q) cases")
        print(f"mass-weighted share where linearized and scanned widths agree within a factor of two: {share:.3f}")
        for lab, m_ in [("both zero", zero_l & zero_s), ("linear zero, scan > 0", zero_l & ~zero_s), ("scan zero, linear > 0", zero_s & ~zero_l),
                        ("both finite > 0", both_fin), ("either full range", inf_l | inf_s)]:
            print(f"  {lab:24s} mass share {(mass * m_).sum() / mass.sum():.3f}")
        if both_fin.any():
            r = ratio[both_fin]
            wm = mass[both_fin] / mass[both_fin].sum()
            o = np.argsort(r)
            cw = np.cumsum(wm[o])
            print("  both finite: mass-weighted ratio linear/scan quantiles 10/50/90%:",
                  [round(float(r[o][np.searchsorted(cw, qq)]), 3) for qq in (0.1, 0.5, 0.9)])
        print("KILL" if share < 0.5 else "PASS", "(kill if share < 0.5)")
        np.savez("/data/output/research/algzoo/20261016_run2/reports/neartie_check.npz", rows=rows, w=w, ws=ws, mass=mass)
    elif sys.argv[1] == "score":
        FULL_RANGE = sys.argv[2] if len(sys.argv) > 2 else "inf"
        print("full-range handling:", FULL_RANGE)
        names, acc, models = load_bench(("mult", "zoo", "example"))
        A = dict(zip(names, acc))
        print("trained models (primary check):")
        for nm, p in [("example", example())] + [(f"zoo_seed{s}", zoo(16, 10, s)) for s in range(5)]:
            e, ops = estimate(p)
            print(f"  {nm:10s} true {A[nm]:.4f} estimate {e:.4f} error {100 * (e - A[nm]):+.1f} points; cost {ops / 2880:.0f} samples", flush=True)
        names, acc, models = load_bench(("mult",))
        sig = np.array([float(str(n).split("_")[0][4:]) for n in names])
        E = np.array([estimate(m) for m in models])
        e, ops = E[:, 0], E[:, 1]
        pq = acc * (1 - acc)
        print("noisy copies (diagnostic):")
        for s in (0.005, 0.01, 0.02):
            k = sig == s
            B = ops[k].mean() / 2880
            print(f"  σ={s}: RMSE {np.sqrt(np.mean((e[k] - acc[k])**2)):.4f} bias {np.mean(e[k] - acc[k]):+.4f}; cost {B:.0f} samples;"
                  f" sampling RMSE at that cost {np.sqrt(pq[k].mean() / B):.4f}", flush=True)
        k = acc >= 0.6
        print(f"  copies with accuracy >= 0.6 ({k.sum()}): RMSE {np.sqrt(np.mean((e[k] - acc[k])**2)):.4f}")
        np.savez(f"/data/output/research/algzoo/20261016_run2/reports/neartie_score_{FULL_RANGE}.npz", names=names, est=e, ops=ops, acc=acc)
