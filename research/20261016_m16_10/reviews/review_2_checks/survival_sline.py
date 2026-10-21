"""Review 2 check: how good can the survival estimator be if each low input's failure set is resolved
along the line s = x / r exactly (emulated here by a dense grid), leaving only theta coarse?

Homogeneity: with m1 = r cos(theta), m2 = r sin(theta), x = r s, the network's answer depends on (theta, s)
only. So for one (pair, q, theta) the failure set is a set of s-intervals, and for EVERY radius r
    f_q(r, theta) = sum over failure intervals [Phi(r b) - Phi(r a)] / Phi(r sin theta)
costs no network evaluations. A real estimator would find the interval ends by 1-D piecewise-linear tracking
(as exact_z2 does in theta); here a dense phi = atan(s) grid stands in for it, and the number of linear pieces
along the grid gives the tracking cost.
Survival rule: c0(theta) * prod_q (1 - f_q), q over low positions after the first spike (or all).
Only theta uses Q_T equal-probability nodes. Nothing is fitted to the copies.
"""
import sys

import numpy as np
import torch as th
from scipy.special import ndtr

from algzoo.bench import load_bench
from algzoo.core import example
from algzoo.exact_z2 import HI, LO, g_angle

dev = "cuda"
PAIRS = [(a, b) for a in range(10) for b in range(10) if a != b]
G = int(sys.argv[2]) if len(sys.argv) > 2 else 1024
_r, _wr = np.polynomial.legendre.leggauss(160)
R = 6.0 * (_r + 1)
WR = 6.0 * _wr


def theta_nodes(qt):
    t = np.linspace(LO, HI, 4001)
    tm = 0.5 * (t[1:] + t[:-1])
    p = g_angle(tm)
    p = p / p.sum()
    c = np.cumsum(p)
    edges = np.searchsorted(c, np.linspace(0, 1, qt + 1)[1:-1])
    parts = np.split(np.arange(len(tm)), edges)
    return np.array([(p[i] * tm[i]).sum() / p[i].sum() for i in parts]), np.array([p[i].sum() for i in parts])


def build(theta, all_q):
    """Sequences for one theta: per pair, one two-spike sequence and G grid points for each low position q."""
    st = np.sin(theta)
    phi_hi = np.arctan(st)
    phi = np.linspace(-np.pi / 2, phi_hi, G + 1)
    phic = 0.5 * (phi[1:] + phi[:-1])                     # grid points (cell centres)
    s_edges = np.tan(phi)
    s_edges[0] = -np.inf
    rows, meta = [], []
    for pi, (p1, p2) in enumerate(PAIRS):
        base = np.zeros(10)
        base[p1], base[p2] = np.cos(theta), st
        rows.append(base[None])
        meta.append((pi, -1))
        first = min(p1, p2)
        for q in range(10):
            if q in (p1, p2) or (not all_q and q < first):
                continue
            X = np.repeat(base[None], G, 0)
            X[:, q] = np.tan(phic)
            rows.append(X)
            meta.append((pi, q))
    return np.concatenate(rows), meta, s_edges


@th.no_grad()
def run(params, X):
    w_hi, w_hh, w_oh = (th.as_tensor(p, device=dev, dtype=th.float64) for p in params)
    x = th.as_tensor(X, device=dev)
    h = th.zeros(x.shape[0], 16, device=dev, dtype=th.float64)
    pats = []
    for t in range(10):
        pre = h @ w_hh.T + x[:, t, None] * w_hi
        pats.append((pre > 0).to(th.int32))
        h = th.relu(pre)
    pred = (h @ w_oh.T).argmax(-1).cpu().numpy()
    bits = th.stack(pats, 1)                               # [N, 10, 16]
    code = (bits * (2 ** th.arange(16, device=dev, dtype=th.int32))).sum(-1).cpu().numpy()   # [N, 10]
    return pred, code


def estimate(params, qt, all_q=False, count=False):
    tn, tw = theta_nodes(qt)
    total, ops = 0.0, 0
    for theta, wt in zip(tn, tw):
        X, meta, s_edges = build(theta, all_q)
        pred, code = run(params, X)
        wr = R * np.exp(-R**2 / 2) * ndtr(R * np.sin(theta)) ** 8 * WR
        wr = wr / wr.sum()
        Phi_edges = ndtr(R[:, None] * s_edges[None, :])     # [nr, G+1]
        Phi_top = ndtr(R * np.sin(theta))
        pos = 0
        surv_pairs = np.zeros(len(PAIRS))
        groups = {}
        cur = None
        for pi, q in meta:
            p2 = PAIRS[pi][1]
            if q == -1:
                c0 = pred[pos] == p2
                pos += 1
                cur = [pi, c0, np.ones(len(R))]
                groups[pi] = cur
                continue
            wrong = pred[pos:pos + G] != p2
            cd = code[pos:pos + G]
            pos += G
            f = ((Phi_edges[:, 1:] - Phi_edges[:, :-1]) * wrong[None, :]).sum(1) / Phi_top
            cur[2] *= np.clip(1 - f, 0, 1)
            if count:                                          # tracking cost from step q on, 2-column pieces
                p1 = PAIRS[pi][0]
                for t in range(q, 10):
                    hist = cd[:, : t + 1]
                    pieces = 1 + int((np.any(hist[1:] != hist[:-1], axis=1)).sum())
                    ops += pieces * (16 * 16 * 2 + 16 * 4)
                ops += pieces * 10 * 16 * 2
        for pi, (p1, p2) in enumerate(PAIRS):
            _, c0, sr = groups[pi]
            surv_pairs[pi] = c0 * (wr * sr).sum()
            if count:
                ops += (10 - min(p1, p2)) * 272 + 160          # shared two-spike run
        total += wt * surv_pairs.mean()
    return total, ops


if __name__ == "__main__":
    qt = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    all_q = "--all" in sys.argv
    b, ops = estimate(example(), qt, all_q, count=True)
    print(f"theta nodes {qt}, grid {G}, all_q {all_q}: base {b:.4f} (true 0.9531); tracking cost without time-shift sharing {ops:.3g} = {ops / 2880:.0f} samples")
    names, acc, models = load_bench(("mult",))
    sig = np.array([float(str(n).split("_")[0][4:]) for n in names])
    est = np.array([estimate(m, qt, all_q)[0] for m in models])
    np.save(f"survival_sline_{qt}_{G}_{int(all_q)}.npy", est)
    for lab, e in (("survival s-line", est), ("x base ratio", est * 0.9531 / b)):
        out = [f"{lab:18s} RMSE {np.sqrt(np.mean((e - acc) ** 2)):.4f} bias {np.mean(e - acc):+.4f} corr {np.corrcoef(e, acc)[0, 1]:.3f}"]
        for s in (0.005, 0.01, 0.02):
            q = sig == s
            out.append(f"σ={s}: {np.sqrt(np.mean((e[q] - acc[q]) ** 2)):.4f} ({np.mean(e[q] - acc[q]):+.3f})")
        print(" | ".join(out), flush=True)
