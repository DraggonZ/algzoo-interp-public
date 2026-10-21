"""Cheap two-spike accuracy: 18 basic runs at K angles, boundaries by linear interpolation of the margin.

The basic run (gap, order) at angle theta gives the final scores of every pair with that gap and order (time shift).
Margin = score of the second's slot minus the best other. Correct angles are where the margin is positive; between grid
angles the boundary is placed by linear interpolation. The angular weight g(theta) is a fixed table (it does not depend on
the model). Cost counts the network steps and readouts actually done.
"""
import sys

import numpy as np

from algzoo.exact_z2 import LO, HI, N_SEQ, angle_mass


def grid(K):
    # denser near ratio 1 (theta -> pi/4) and near a small second (theta ~ 0); cover the negative-second range coarsely
    base = np.r_[np.linspace(LO + 0.05, -0.2, max(2, K // 4)), np.linspace(-0.1, 0.35, max(2, K // 4)),
                 np.linspace(0.45, HI - 0.004, K - 2 * max(2, K // 4))]
    return np.sort(base)


MASS = {}


def mass(a, b):
    key = (round(a, 12), round(b, 12))
    if key not in MASS:
        MASS[key] = angle_mass(a, b)
    return MASS[key]


def z2_cheap(params, K=8):
    w_hi, w_hh, w_oh = (np.asarray(p, np.float64) for p in params)
    d = len(w_hi)
    th = grid(K)
    U = np.c_[np.cos(th), np.sin(th)]             # [K, 2]: (largest, second) values
    ops = 0
    total = 0.0
    for gap in range(1, N_SEQ):
        for max_first in (True, False):
            h = np.zeros((K, d))
            for t in range(N_SEQ):
                x = np.zeros(K)
                if t == 0:
                    x = U[:, 0] if max_first else U[:, 1]
                elif t == gap:
                    x = U[:, 1] if max_first else U[:, 0]
                h = np.maximum(0, h @ w_hh.T + x[:, None] * w_hi)
                ops += K * (d * d + d)
                if t >= gap:
                    s = N_SEQ - 1 - t
                    p1, p2 = (s, s + gap) if max_first else (s + gap, s)
                    L = h @ w_oh.T; ops += K * w_oh.size
                    Lo = L.copy(); Lo[:, p2] = -np.inf
                    m = L[:, p2] - Lo.max(1)
                    # boundaries by interpolation; ends extend to LO and HI with the end sign
                    edges = [LO]; signs = [m[0] > 0]
                    for k in range(K - 1):
                        if (m[k] > 0) != (m[k + 1] > 0):
                            edges.append(th[k] + (th[k + 1] - th[k]) * m[k] / (m[k] - m[k + 1]))
                            signs.append(m[k + 1] > 0)
                    edges.append(HI)
                    for i, sg in enumerate(signs):
                        if sg:
                            total += mass(edges[i], edges[i + 1])
    return total, ops


if __name__ == "__main__":
    from algzoo.bench import load_bench
    from algzoo.core import example, zoo
    from algzoo.exact_z2 import exact_z2
    Z = np.load("/data/output/research/algzoo/20261016_run2/reports/z2_check.npz", allow_pickle=True)
    z2 = dict(zip([str(n) for n in Z["names"]], Z["z2_exact"]))
    for K in [int(a) for a in sys.argv[1:]] or [4, 8, 16]:
        errs = []
        for name in ["example", 0, 1, 2, 3, 4]:
            p = example() if name == "example" else zoo(16, 10, name)
            zc, ops = z2_cheap(p, K)
            ze = exact_z2(p)[0]
            errs.append(zc - ze)
        names, acc, models = load_bench(("mult",))
        ce = [z2_cheap(m, K)[0] - z2[str(n)] for n, m in zip(names, models)]
        print(f"K={K}: cost {ops/2880:.0f} samples' worth; trained models error {np.round(errs, 4)}; copies RMSE {np.sqrt(np.mean(np.square(ce))):.4f}, bias {np.mean(ce):+.4f}")
