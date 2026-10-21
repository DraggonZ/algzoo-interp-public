"""How does a model's mean unsigned second-third tie bias depend on the largest's value (relative to the
tied pair at 1.0)? And what is the distribution of largest / second at second-third near-ties?"""
import itertools

import numpy as np
import torch as th

from algzoo.core import example, forward, zoo

pl = np.array(list(itertools.permutations(range(10), 3))); N = len(pl); r = np.arange(N)


def w23(P, m1, bg=-0.46):
    base = np.full((N, 10), bg); base[r, pl[:, 0]] = m1; base[r, pl[:, 1]] = 1.0
    lo, hi = np.full(N, 0.8), np.full(N, 1.2)
    for _ in range(22):
        mid = (lo + hi) / 2; x = base.copy(); x[r, pl[:, 2]] = mid
        s = forward(P, th.tensor(x, device="cuda")).cpu().numpy()
        f = s[r, pl[:, 2]] > s[r, pl[:, 1]]
        hi = np.where(f, mid, hi); lo = np.where(f, lo, mid)
    return np.mean(np.abs((lo + hi) / 2 - 1))


rng = np.random.default_rng(1)
x = np.sort(rng.standard_normal((5_000_000, 10)), 1)
m1, m2, m3 = x[:, -1], x[:, -2], x[:, -3]
tie = (m2 > 0) & (m2 - m3 < 0.01 * m2)
ratio = m1[tie] / m2[tie]
qs = np.percentile(ratio, [10, 25, 50, 75, 90])
print("largest / second at second-third near-ties: percentiles 10/25/50/75/90:", qs.round(2))
# the background: mean of the other seven at those ties, relative to the second
bgr = x[tie][:, :7].mean(1) / m2[tie]
print("mean of the seven smaller numbers / second at those ties: median", np.median(bgr).round(2))
grid = [1.05, 1.15, 1.3, 1.6, 2.0, 3.0, 5.0]
for name, P in (("ARC", example()), ("zoo0", zoo(16, 10, 0)), ("zoo2", zoo(16, 10, 2))):
    ws = [w23(P, g) for g in grid]
    # weight by the near-tie distribution of the ratio
    edges = np.r_[1.0, [(a + b) / 2 for a, b in zip(grid[:-1], grid[1:])], np.inf]
    wts = np.array([np.mean((ratio >= a) & (ratio < b)) for a, b in zip(edges[:-1], edges[1:])])
    print(f"{name}: mean |bias| at largest = {grid}: {np.round(ws, 4)}; weighted by the near-tie distribution: {np.dot(wts, ws):.4f} (at 1.6: {ws[3]:.4f})")
