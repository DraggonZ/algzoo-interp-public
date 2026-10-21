"""What is the gap between a clean network's decision quantities and ARC's made of? Gap E = ARC's (D_T, D_L) minus the
clean network's, in votes, on lists without huge suppression terms. Share of E's variance explained by: (1) one linear
rule in (x_L, x_S, x_T) per arrangement (fixable by matching typical behaviour); (2) the same per arrangement and ratio
bin; (3) plus kept-number content (positive parts of the other seven by position). Held out. v21, v19 kappa 0.5, v7z."""
import sys, numpy as np, torch
sys.path.insert(0, "../../family"); sys.path.insert(0, "..")
from common import *
from algzoo.handbuilt_v21 import build_v21
from algzoo.handbuilt_v19 import build_v19
from baseline_v7z import nets
from shuffle_gap import vote_unit
def cellfit(Fe, y, cell, ncell, tr, te):
    A = torch.zeros(ncell, Fe.shape[1], Fe.shape[1], device=DEV, dtype=dt); B = torch.zeros(ncell, Fe.shape[1], y.shape[1], device=DEV, dtype=dt)
    A.index_add_(0, cell[tr], Fe[tr, :, None] * Fe[tr, None, :]); B.index_add_(0, cell[tr], Fe[tr, :, None] * y[tr, None, :])
    W = torch.linalg.solve(A + 1e-6 * torch.eye(Fe.shape[1], device=DEV, dtype=dt), B); P = (Fe[te, :, None] * W[cell[te]]).sum(1)
    return 1 - ((P - y[te]) ** 2).sum() / ((y[te] - y[te].mean(0)) ** 2).sum()
if __name__ == "__main__":
    g = torch.Generator(device=DEV).manual_seed(203); X = torch.randn(2000000, 10, device=DEV, dtype=dt, generator=g)
    o = X.argsort(1, descending=True); L, S, Tq = o[:, 0], o[:, 1], o[:, 2]; ar = torch.arange(len(X), device=DEV)
    v = torch.stack([X[ar, L], X[ar, S], X[ar, Tq]], 1); xs = v[:, 1]
    rb = torch.bucketize(v[:, 0] / xs.abs().clamp(min=1e-3), torch.tensor([1.1, 1.25, 1.5, 2.0, 3.0], device=DEV, dtype=dt))
    c1 = L * 100 + S * 10 + Tq; c2 = c1 * 6 + rb
    rest = X.clamp(min=0).clone()
    for p in (L, S, Tq): rest[ar, p] = 0
    def D(W):
        u = vote_unit(W, X[:200000]); Sc = scores(W, X) / u; return torch.stack([Sc[ar, Tq] - Sc[ar, S], Sc[ar, L] - Sc[ar, S]], 1)
    Da = D(ARC); half = len(X) // 2
    for nm, W in (("v21", T(build_v21())), ("v19 kappa 0.5", T(build_v19(kappa=0.5).params)), ("v7z", nets()["v7 + z + ninth slot (18)"])):
        Dc = D(W); E = Da - Dc; ok = ~(Dc.abs() > 50).any(1)
        idx = torch.nonzero(ok)[:, 0]; tr, te = idx[idx < half], idx[idx >= half]
        r1 = cellfit(v, E, c1, 1000, tr, te); r2 = cellfit(v, E, c2, 6000, tr, te); r3 = cellfit(torch.cat([v, rest], 1), E, c1, 1000, tr, te)
        print(f"{nm:16s} gap RMS {E[ok].pow(2).mean().sqrt():.2f} (ARC's D RMS {Da[ok].pow(2).mean().sqrt():.2f}); share of the gap explained: per-arrangement linear {100*r1:.0f}%, + ratio bins {100*r2:.0f}%, per-arrangement linear incl. kept numbers {100*r3:.0f}%", flush=True)
