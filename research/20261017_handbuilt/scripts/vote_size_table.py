"""ARC: vote size of the second, per unit, in the reduced setting (largest 1.0 at p1, second
r*1.0 at p2, the seven others at background b). Vote = d(score_p2 - max other score... ) -> use
d(score_p2 - mean)/d x_p2. Tables by (p1, p2). Then test whether it factors:
second after largest: f(p2 - p1) * g(9 - p2); second before: f'(p1 - p2) * g'(9 - p1) etc."""
import itertools

import numpy as np
import torch as th

from algzoo.core import example, forward

P = example()
for bg in (-0.46, 0.0):
    for r in (0.9,):
        xs = []
        pairs = [(p1, p2) for p1, p2 in itertools.permutations(range(10), 2)]
        for p1, p2 in pairs:
            x = np.full(10, bg); x[p1] = 1.0; x[p2] = r
            xs.append(x)
        x = th.tensor(np.array(xs), requires_grad=True)
        s = forward(P, x)
        sc = s - s.mean(-1, keepdim=True)
        idx = th.tensor([p2 for _, p2 in pairs])
        g = th.autograd.grad(sc[th.arange(len(pairs)), idx].sum(), x)[0]
        V = np.full((10, 10), np.nan)
        for k, (p1, p2) in enumerate(pairs):
            V[p1, p2] = g[k, p2].item()
        print(f"\nbackground {bg}, ratio {r}: vote of the second per unit (rows p1, cols p2)")
        print(np.array2string(V, precision=0, max_line_width=200))
        for lab, cond, a_of, b_of in (("second after largest", lambda p1, p2: p2 > p1, lambda p1, p2: p2 - p1, lambda p1, p2: 9 - p2),
                                       ("second before largest", lambda p1, p2: p2 < p1, lambda p1, p2: p1 - p2, lambda p1, p2: 9 - p1)):
            cells = [(a_of(p1, p2), b_of(p1, p2), V[p1, p2]) for p1, p2 in pairs if cond(p1, p2)]
            A = np.full((10, 10), np.nan)
            for a, b, v in cells:
                A[a, b] = v
            # rank-1 fit on observed cells by alternating least squares (log-free, values positive?)
            f = np.nanmean(A, 1); f = np.nan_to_num(f, nan=1.0); gg = np.ones(10)
            for _ in range(200):
                gg = np.array([np.nansum(A[:, j] * f) / np.nansum(np.where(np.isnan(A[:, j]), 0, f ** 2)) if np.any(~np.isnan(A[:, j])) else 0 for j in range(10)])
                f = np.array([np.nansum(A[i] * gg) / np.nansum(np.where(np.isnan(A[i]), 0, gg ** 2)) if np.any(~np.isnan(A[i])) else 0 for i in range(10)])
            pred = np.outer(f, gg)
            m = ~np.isnan(A)
            r2 = 1 - np.nanmean((A - pred)[m] ** 2) / np.nanvar(A[m])
            add = np.nanmean(A, 1)[:, None] + np.nanmean(A, 0)[None] - np.nanmean(A)
            r2a = 1 - np.nanmean((A - add)[m] ** 2) / np.nanvar(A[m])
            print(f"  {lab}: {m.sum()} cells, spread (sd) {np.nanstd(A[m]):.1f} around mean {np.nanmean(A[m]):.1f}; "
                  f"product rule R^2 {r2:.2f}; additive rule R^2 {r2a:.2f}")
