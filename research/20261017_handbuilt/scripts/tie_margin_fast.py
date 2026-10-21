"""Close-call bias from one run plus a derivative per placement, instead of bisection.
At the tie (third = second = 1), margin M = s[p2] - s[p3] and slope S = dM/d(third). Inside one
linear region, the flip point is 1 - M/S. Compare with bisection for ARC's model."""
import itertools

import numpy as np
import torch as th

from algzoo.core import example, forward

pl = np.array(list(itertools.permutations(range(10), 3))); N = len(pl); r = np.arange(N)


def fast_b23(P, m1=1.6, bg=-0.46, eps=1e-6):
    x = np.full((N, 10), bg); x[r, pl[:, 0]] = m1; x[r, pl[:, 1]] = 1.0; x[r, pl[:, 2]] = 1.0 - eps
    X = th.tensor(x, device="cuda", requires_grad=True)
    s = forward(P, X)
    M = s[r, pl[:, 1]] - s[r, pl[:, 2]]
    g = th.autograd.grad(M.sum(), X)[0][r, pl[:, 2]]
    return (-(M / g)).detach().cpu().numpy()  # third must rise by -M/S above 1 to flip -> b = -M/S


def bisect_b23(P, m1=1.6, bg=-0.46):
    base = np.full((N, 10), bg); base[r, pl[:, 0]] = m1; base[r, pl[:, 1]] = 1.0
    lo, hi = np.full(N, 0.8), np.full(N, 1.2)
    for _ in range(26):
        mid = (lo + hi) / 2; x = base.copy(); x[r, pl[:, 2]] = mid
        s = forward(P, th.tensor(x, device="cuda")).cpu().numpy()
        f = s[r, pl[:, 2]] > s[r, pl[:, 1]]
        hi = np.where(f, mid, hi); lo = np.where(f, lo, mid)
    return (lo + hi) / 2 - 1


P = example()
bf, bb = fast_b23(P), bisect_b23(P)
d = np.abs(bf - bb)
print(f"fast vs bisection: median |difference| {np.median(d):.5f}, share within 0.002: {np.mean(d < 0.002):.3f}, max {d.max():.3f}")
print(f"one-sided mean: fast {np.mean(np.maximum(0, -bf)):.4f}, bisection {np.mean(np.maximum(0, -bb)):.4f}")


def newton_b23(P, steps, m1=1.6, bg=-0.46):
    """Newton steps on the piecewise-linear margin, starting at the tie; clipped to [-0.2, 0.2]."""
    v = np.full(N, 1.0 - 1e-6)
    for _ in range(steps):
        x = np.full((N, 10), bg); x[r, pl[:, 0]] = m1; x[r, pl[:, 1]] = 1.0; x[r, pl[:, 2]] = v
        X = th.tensor(x, device="cuda", requires_grad=True)
        s = forward(P, X); M = s[r, pl[:, 1]] - s[r, pl[:, 2]]
        g = th.autograd.grad(M.sum(), X)[0][r, pl[:, 2]]
        step = (-(M / g)).detach().cpu().numpy()
        v = np.clip(v + np.nan_to_num(step, nan=0.0, posinf=0.2, neginf=-0.2), 0.8, 1.2)
    return v - 1


for k in (1, 2, 3, 4):
    bn = newton_b23(P, k)
    d = np.abs(bn - bb)
    print(f"{k} run(s): within 0.002 of bisection {np.mean(d < 0.002):.3f}; one-sided mean {np.mean(np.maximum(0, -bn)):.4f} (bisection {np.mean(np.maximum(0, -bb)):.4f})")
