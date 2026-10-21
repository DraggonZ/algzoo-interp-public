"""Shared helpers for the family-mistakes study: networks, exact per-list linear maps (no biases, piecewise linear, so
scores = J x exactly on each list), near-tie random lists."""
import numpy as np, torch
from algzoo.core import load, example
DEV = "cuda"; dt = torch.float64
def T(W): return [torch.tensor(np.asarray(p, float), device=DEV, dtype=dt) for p in W]
def P(x): return x.params if hasattr(x, "params") else x
def trained(s): return T(load(f"/data/output/research/algzoo/20261016_run3/models/2nd_argmax_16_10_1073741824_{s}.pth"))
ARC = T(example())
def scores(W, X):
    h = torch.zeros(len(X), len(W[0]), device=DEV, dtype=dt)
    for t in range(10): h = torch.relu(h @ W[1].T + X[:, t:t + 1] * W[0])
    return h @ W[2].T
def jac(W, X):
    """J[n, out, t] with scores = J @ x exactly."""
    X = X.clone().requires_grad_(True); S = scores(W, X); J = []
    for j in range(10):
        g, = torch.autograd.grad(S[:, j].sum(), X, retain_graph=j < 9); J.append(g)
    return torch.stack(J, 1).detach()
def near_lists(n, gap=0.05, seed=0, positive=True):
    """Random lists where the third is within `gap` (relative) of the second; second positive if asked."""
    g = torch.Generator(device=DEV).manual_seed(seed); out = []
    while sum(len(o) for o in out) < n:
        X = torch.randn(400000, 10, device=DEV, dtype=dt, generator=g); v = X.sort(1, descending=True).values
        m = (v[:, 1] - v[:, 2]) < gap * v[:, 1].abs()
        if positive: m &= v[:, 1] > 0
        out.append(X[m])
    return torch.cat(out)[:n]
def top3(X):
    o = X.argsort(1, descending=True); return o[:, 0], o[:, 1], o[:, 2]
