"""How compressible is ARC's decision among the three largest? For each cell of a partition of the arrangements
(positions of L, S, T), a linear rule: score_T - score_S and score_L - score_S as a x_L + b x_S + c x_T (no intercept;
the network has no biases). Partitions: arrival order (6 cells); order and L's position; order and all three offsets
from L... ; every exact arrangement (720). Fit on 2,000,000 random lists (analysis fit), scored as a stand-alone scorer
over the three largest only (it cannot name another position) by ARC's mistakes reproduced and near-tie behaviour."""
import numpy as np, torch
from common import *
import score
def cells(X, kind):
    o = X.argsort(1, descending=True); L, S, Tq = o[:, 0], o[:, 1], o[:, 2]
    order = (L < S).long() * 4 + (L < Tq).long() * 2 + (S < Tq).long()          # encodes the arrival order
    if kind == "order": return order
    if kind == "order + L position": return order * 10 + L
    if kind == "order + L and S positions": return (order * 10 + L) * 10 + S
    return (L * 10 + S) * 10 + Tq
NC = {"order": 8, "order + L position": 80, "order + L and S positions": 800, "exact arrangement": 1000}
def fit(kind, n=2000000):
    g = torch.Generator(device=DEV).manual_seed(31); W = torch.zeros(NC[kind], 2, 3, device=DEV, dtype=dt); cnt = 0
    A = torch.zeros(NC[kind], 3, 3, device=DEV, dtype=dt); B = torch.zeros(NC[kind], 3, 2, device=DEV, dtype=dt)
    for _ in range(n // 500000):
        X = torch.randn(500000, 10, device=DEV, dtype=dt, generator=g); o = X.argsort(1, descending=True); ar = torch.arange(len(X), device=DEV)
        v = torch.stack([X[ar, o[:, k]] for k in range(3)], 1); Sc = scores(ARC, X)
        y = torch.stack([Sc[ar, o[:, 2]] - Sc[ar, o[:, 1]], Sc[ar, o[:, 0]] - Sc[ar, o[:, 1]]], 1)
        c = cells(X, kind)
        A.index_add_(0, c, v[:, :, None] * v[:, None, :]); B.index_add_(0, c, v[:, :, None] * y[:, None, :])
    sol = torch.linalg.solve(A + 1e-6 * torch.eye(3, device=DEV, dtype=dt), B)    # [cells, 3, 2]
    return sol
def scorer(kind):
    sol = fit(kind)
    def f(X):
        out = torch.full((len(X), 10), -1e9, device=DEV, dtype=dt); o = X.argsort(1, descending=True); ar = torch.arange(len(X), device=DEV)
        v = torch.stack([X[ar, o[:, k]] for k in range(3)], 1); w = sol[cells(X, kind)]
        d = (v[:, :, None] * w).sum(1)
        out[ar, o[:, 1]] = 0; out[ar, o[:, 2]] = d[:, 0]; out[ar, o[:, 0]] = d[:, 1]
        return out
    return f, sol
if __name__ == "__main__":
    models = {}
    for kind in NC:
        f, sol = scorer(kind); models[f"{kind} ({(sol.abs().sum((1,2)) > 0).sum().item() * 6} numbers)"] = [f]
    models["ARC itself"] = [ARC]
    score.report(models)
