"""Review 2 prototype: a deterministic 'independent single-input failure' estimator on a coarse grid.

Rule (exact identity under the model's own input law): given the positions (p1, p2) and values (m1, m2) of the
top two, the 8 low inputs are iid N(0,1) truncated to (-inf, m2). So
    P(two-spike answer right and no single low input alone breaks it)
      = sum_pairs E_{m1,m2}[ 1(Z2 right) * prod_p (1 - f_p(m1, m2)) ],
with f_p = P_x(argmax L(m1 e_p1 + m2 e_p2 + x e_p) != p2). This 'indep' quantity tracked the copies with RMSE
0.024 in the sampled diagnostic (combine_rules.py). Here every expectation is a deterministic quantile-midpoint
rule: Q2 nodes for m2 (2nd-largest-of-10 law), Q1 for m1 | m2, QX for x | m2. No random numbers.
Ops: one step (272) per distinct leading-zero-stripped input prefix, plus 160 per distinct readout. This uses
only prefix sharing and the fact that leading zeros leave h = 0, both deterministic.
"""
import sys

import numpy as np
import torch as th
from scipy.optimize import brentq
from scipy.special import ndtr, ndtri

from algzoo.bench import load_bench
from algzoo.core import example, forward

BASE_ACC = 0.9531
n = 10


def F2(m):                       # CDF of the 2nd largest of 10 iid N(0,1)
    P = ndtr(m)
    return P**10 + 10 * P**9 * (1 - P)


def grid(Q2, Q1, QX):
    m2s = [brentq(lambda m: F2(m) - (j + 0.5) / Q2, -6, 6) for j in range(Q2)]
    nodes = []
    for m2 in m2s:
        P2 = ndtr(m2)
        for i in range(Q1):
            m1 = ndtri(P2 + (i + 0.5) / Q1 * (1 - P2))
            xs = ndtri((np.arange(QX) + 0.5) / QX * P2)
            nodes.append((m1, m2, xs))
    return nodes


def build(Q2, Q1, QX):
    nodes = grid(Q2, Q1, QX)
    seqs, meta = [], []           # meta: (pair index, node index, low position or -1)
    pairs = [(a, b) for a in range(n) for b in range(n) if a != b]
    for pi, (p1, p2) in enumerate(pairs):
        for ni, (m1, m2, xs) in enumerate(nodes):
            base = np.zeros(n)
            base[p1], base[p2] = m1, m2
            seqs.append(base.copy())
            meta.append((pi, ni, -1))
            for p in range(n):
                if p in (p1, p2):
                    continue
                for x in xs:
                    s = base.copy()
                    s[p] = x
                    seqs.append(s)
                    meta.append((pi, ni, p))
    return np.array(seqs), np.array(meta), pairs, len(nodes), QX


def count_ops(seqs):
    prefixes, finals = set(), set()
    for s in seqs:
        nz = np.flatnonzero(s)
        t0 = nz[0]
        for t in range(t0, n):
            prefixes.add(tuple(s[t0:t + 1]))
        finals.add(tuple(s[t0:]))
    return len(prefixes) * 272 + len(finals) * 160, len(seqs) * 2880


@th.no_grad()
def estimate(params, seqs, meta, pairs, n_nodes, QX):
    x = th.as_tensor(seqs, device="cuda")
    pred = forward(params, x).argmax(-1).cpu().numpy()
    p2 = np.array([pairs[pi][1] for pi in meta[:, 0]])
    ok = pred == p2
    tot = 0.0
    z2 = 0.0
    # group by (pair, node)
    key = meta[:, 0] * n_nodes + meta[:, 1]
    for k in np.unique(key):
        sel = key == k
        m = meta[sel]
        o = ok[sel]
        zr = o[m[:, 2] == -1][0]
        z2 += zr
        if not zr:
            continue
        prod = 1.0
        for p in np.unique(m[:, 2]):
            if p < 0:
                continue
            prod *= o[m[:, 2] == p].mean()          # 1 - f_p
        tot += prod
    return tot / (len(pairs) * n_nodes), z2 / (len(pairs) * n_nodes)


if __name__ == "__main__":
    Q2, Q1, QX = (int(a) for a in sys.argv[1:4])
    seqs, meta, pairs, nn, qx = build(Q2, Q1, QX)
    ops, naive = count_ops(seqs)
    print(f"grid Q2={Q2} Q1={Q1} QX={QX}: {len(seqs)} sequences; ops with prefix sharing {ops:.3g} = {ops / 2880:.0f} samples (naive {naive / 2880:.0f})")
    b, bz = estimate(example(), seqs, meta, pairs, nn, qx)
    print(f"base: indep {b:.4f}  Z2 on grid {bz:.4f}  (true 0.9531)")
    names, acc, models = load_bench(("mult",))
    sig = np.array([float(str(nm).split("_")[0][4:]) for nm in names])
    est = np.array([estimate(m, seqs, meta, pairs, nn, qx)[0] for m in models])
    samp = np.load("combine_rules.npz")
    keys = list(samp["keys"])
    ind_s = samp["R"][:, keys.index("indep")]
    C = ops / 2880
    need = 0.675 / np.sqrt(max(C, 100))
    for lab, e in (("grid indep", est), ("grid indep x base ratio", est * BASE_ACC / b)):
        out = [f"{lab:24s} RMSE {np.sqrt(np.mean((e - acc) ** 2)):.4f} bias {np.mean(e - acc):+.4f} corr {np.corrcoef(e, acc)[0, 1]:.3f}"]
        for s in (0.005, 0.01, 0.02):
            q = sig == s
            out.append(f"σ={s}: {np.sqrt(np.mean((e[q] - acc[q]) ** 2)):.4f}")
        print(" | ".join(out))
    print(f"grid vs sampled indep rule: RMSE {np.sqrt(np.mean((est - ind_s) ** 2)):.4f}")
    samp_rmse = np.sqrt(np.mean(acc * (1 - acc)) / max(C, 1))
    print(f"milestone at this cost needs RMSE <= {need:.4f}; sampling RMSE at this cost {samp_rmse:.4f}")
    np.save(f"indep_grid_{Q2}_{Q1}_{QX}.npy", est)
