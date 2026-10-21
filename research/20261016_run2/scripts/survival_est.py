"""Deterministic 'survival' estimator of M16,10 accuracy (no sampled inputs).

Structure used: the positions (p1, p2) of the top two are uniform over the 90 ordered pairs; given the top two values,
the 8 other inputs are independent N(0,1) conditioned on being below m2. The surrogate counts an input as right
when the network is right with only the top two nonzero (c0) AND right with the top two plus each single low input
that arrives after the first spike (others zero). Its expectation per pair is
    E_{m1,m2}[ c0 · Π_q s_q ],   s_q = P_x(right with x at q | m1, m2),   x ~ N(0,1) | x < m2,
and factors over q exactly (no independence assumption beyond the order-statistic structure).
Quadrature: (θ, r) product Gauss nodes for the top-two density of one pair; k equal-probability nodes for each low x.
Cost counted as multiply-adds with prefix sharing (state before the first spike is zero; the two-spike run is shared
by all q; each (q, x-node) reruns from step q to the end).
"""
import sys

import numpy as np
import torch as th
from scipy.special import ndtr, ndtri

from algzoo.bench import load_bench
from algzoo.core import example

dev = "cuda"
LO, HI = -0.75 * np.pi, 0.25 * np.pi
PAIRS = [(a, b) for a in range(10) for b in range(10) if a != b]


def radial_rule(theta, n_r):
    """Gauss nodes in r for density r e^{-r²/2} Φ(r sinθ)^8 on (0, ∞), via a fine grid and moment matching."""
    r = np.linspace(1e-4, 9, 4000)
    w = r * np.exp(-r**2 / 2) * ndtr(r * np.sin(theta)) ** 8
    mass = np.trapezoid(w, r)
    p = w / w.sum()
    if n_r == 1:
        return np.array([(p * r).sum()]), np.array([mass])
    # n_r equal-probability bins, node = bin's conditional mean
    c = np.cumsum(p)
    edges = np.searchsorted(c, np.linspace(0, 1, n_r + 1)[1:-1])
    parts = np.split(np.arange(len(r)), edges)
    nodes = np.array([(p[i] * r[i]).sum() / p[i].sum() for i in parts])
    return nodes, mass * np.array([p[i].sum() for i in parts])


def angle_rule(n_t):
    """Equal-probability bins in θ for the angular density g(θ) (same for every pair); node = conditional mean."""
    t = np.linspace(LO, HI, 20001)[:-1] + (HI - LO) / 40000
    g = np.array([np.trapezoid(rr * np.exp(-rr**2 / 2) * ndtr(rr * np.sin(tt)) ** 8, rr) for tt in t[::20]
                  for rr in [np.linspace(1e-4, 9, 800)]])
    t = t[::20]
    p = g / g.sum()
    c = np.cumsum(p)
    edges = np.searchsorted(c, np.linspace(0, 1, n_t + 1)[1:-1])
    parts = np.split(np.arange(len(t)), edges)
    return np.array([(p[i] * t[i]).sum() / p[i].sum() for i in parts]), np.array([p[i].sum() for i in parts])


def build_nodes(n_t, n_r):
    """(m1, m2) nodes and weights for one pair; weights sum to 1/90."""
    th_nodes, th_w = angle_rule(n_t)
    M, W = [], []
    for t, wt in zip(th_nodes, th_w):
        rn, rw = radial_rule(t, n_r)
        for r, w in zip(rn, rw / rw.sum()):
            M.append((r * np.cos(t), r * np.sin(t)))
            W.append(wt * w / 90)
    return np.array(M), np.array(W)


@th.no_grad()
def estimate(params, M, W, k, all_q=False, rescue=False):
    w_hi, w_hh, w_oh = (th.as_tensor(p, device=dev, dtype=th.float64) for p in params)
    K = len(M)
    u = (np.arange(k) + 0.5) / k                                   # equal-probability x nodes
    total, ops = 0.0, 0
    d = 16
    step = d * d + d                                               # multiply-adds per RNN step
    for p1, p2 in PAIRS:
        first = min(p1, p2)
        qs = [q for q in range(10) if q not in (p1, p2) and (all_q or q > first)]
        m1 = th.as_tensor(M[:, 0], device=dev)
        m2 = th.as_tensor(M[:, 1], device=dev)
        xq = th.as_tensor(ndtri(u[None, :] * ndtr(M[:, 1:2])), device=dev)       # [K, k] x-nodes below m2
        # sequences: [K, 1 + len(qs) * k, 10]
        S = 1 + len(qs) * k
        X = th.zeros(K, S, 10, device=dev, dtype=th.float64)
        X[:, :, p1] = m1[:, None]
        X[:, :, p2] = m2[:, None]
        for j, q in enumerate(qs):
            X[:, 1 + j * k: 1 + (j + 1) * k, q] = xq
        h = th.zeros(K, S, d, device=dev, dtype=th.float64)
        for t in range(10):
            h = th.relu(h @ w_hh.T + X[..., t, None] * w_hi)
        ok = (h @ w_oh.T).argmax(-1) == p2                                      # [K, S]
        c0 = ok[:, 0].double()
        s = th.ones(K, device=dev, dtype=th.float64)
        miss = th.ones(K, device=dev, dtype=th.float64)       # P(no single low input rescues a wrong answer)
        for j in range(len(qs)):
            frac = ok[:, 1 + j * k: 1 + (j + 1) * k].double().mean(1)
            s = s * frac
            miss = miss * (1 - frac)
        val = c0 * s + ((1 - c0) * (1 - miss) if rescue else 0.0)
        total += (th.as_tensor(W, device=dev) * val).sum().item()
        # cost with prefix sharing: two-spike run from the first spike; each (q, node) from q to the end; readouts
        ops += K * ((10 - first) * step + len(qs) * k * 0 + sum((10 - q) * step for q in qs) * k + S * d * 10)
    return total, ops


def main_two_way(cfg=(20, 6, 16)):
    from algzoo.core import zoo
    M, W = build_nodes(*cfg[:2])
    k = cfg[2]
    names, acc, models = load_bench(("mult", "zoo", "example"))
    A = dict(zip(names, acc))
    print("deterministic (dense quadrature), all low positions: model, true, break-only rule, two-way rule")
    for nm, m in [("example", example())] + [(f"zoo_seed{s}", zoo(16, 10, s)) for s in range(5)]:
        b = estimate(m, M, W, k, all_q=True)[0]
        t = estimate(m, M, W, k, all_q=True, rescue=True)[0]
        print(f"  {nm:10s} true {A[nm]:.4f} | break-only {b:.4f} ({b - A[nm]:+.4f}) | two-way {t:.4f} ({t - A[nm]:+.4f})", flush=True)
    names, acc, models = load_bench(("mult",))
    sig = np.array([float(str(n).split("_")[0][4:]) for n in names])
    for lab, kw in [("break-only", {}), ("two-way", {"rescue": True})]:
        e = np.array([estimate(m, M, W, k, all_q=True, **kw)[0] for m in models])
        print(f"  copies {lab:10s} RMSE {np.sqrt(np.mean((e - acc)**2)):.4f} bias {np.mean(e - acc):+.4f} corr {np.corrcoef(e, acc)[0,1]:.3f} | "
              + " ".join(f"σ={s}: {np.sqrt(np.mean((e[sig==s]-acc[sig==s])**2)):.4f}" for s in (0.005, 0.01, 0.02)), flush=True)


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "two_way":
    main_two_way()
elif __name__ == "__main__":
    names, acc, models = load_bench(("mult",))
    sig = np.array([float(str(n).split("_")[0][4:]) for n in names])
    configs = [tuple(int(v) for v in c.split(",")) for c in sys.argv[1:]] or [(40, 8, 32), (3, 2, 8), (2, 1, 4), (1, 1, 4), (1, 1, 2)]
    for n_t, n_r, k in configs:
        M, W = build_nodes(n_t, n_r)
        b, ops = estimate(example(), M, W, k)
        e = np.array([estimate(m, M, W, k)[0] for m in models])
        mse = np.mean((e - acc) ** 2)
        B = ops / 2880
        mse_s = np.mean(acc * (1 - acc)) / B
        print(f"θ nodes {n_t}, r nodes {n_r}, x nodes {k}: base {b:.4f} | RMSE {np.sqrt(mse):.4f} bias {np.mean(e - acc):+.4f} corr {np.corrcoef(e, acc)[0,1]:.3f}"
              f" | " + " ".join(f"σ={s}: {np.sqrt(np.mean((e[sig==s]-acc[sig==s])**2)):.4f}" for s in (0.005, 0.01, 0.02))
              + f" | cost {ops:.3g} = {B:.0f} samples; sampling RMSE {np.sqrt(mse_s):.4f}; MSE ratio {mse / mse_s:.2f}", flush=True)
