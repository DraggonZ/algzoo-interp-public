"""Structure of v19's systematic gap to ARC. Per arrangement, fit E = ARC's (D_T, D_L) minus v19's as a x_L + b x_S + c x_T
(6 coefficients per arrangement). Then explain those coefficients across arrangements with additive factors: positions of
the third, the second and the largest, arrival order, and pairs (third's position x order etc.); share of the
coefficient variance (weighted by how often each arrangement occurs) and of the gap's variance on held-out lists
that each factor model keeps. Also print the main coefficient tables for the strongest factor."""
import sys, itertools, numpy as np, torch
sys.path.insert(0, "../../family"); sys.path.insert(0, "..")
from common import *
from algzoo.handbuilt_v19 import build_v19
from shuffle_gap import vote_unit
from gap_parts import cellfit
if __name__ == "__main__":
    g = torch.Generator(device=DEV).manual_seed(203); X = torch.randn(2000000, 10, device=DEV, dtype=dt, generator=g)
    o = X.argsort(1, descending=True); L, S, Tq = o[:, 0], o[:, 1], o[:, 2]; ar = torch.arange(len(X), device=DEV)
    v = torch.stack([X[ar, L], X[ar, S], X[ar, Tq]], 1)
    def D(W):
        u = vote_unit(W, X[:200000]); Sc = scores(W, X) / u; return torch.stack([Sc[ar, Tq] - Sc[ar, S], Sc[ar, L] - Sc[ar, S]], 1)
    W19 = T(build_v19(kappa=0.5).params); E = D(ARC) - D(W19); ok = ~(D(W19).abs() > 50).any(1)
    order = (L < S).long() * 4 + (L < Tq).long() * 2 + (S < Tq).long()
    factors = {"third's position": Tq, "second's position": S, "largest's position": L, "order": order,
               "third's position x order": Tq * 8 + order, "second's position x order": S * 8 + order, "largest's position x order": L * 8 + order,
               "third x second positions": Tq * 10 + S, "largest x second positions": L * 10 + S}
    idx = torch.nonzero(ok)[:, 0]; half = len(X) // 2; tr, te = idx[idx < half], idx[idx >= half]
    full = cellfit(v, E, L * 100 + S * 10 + Tq, 1000, tr, te)
    print(f"per arrangement (720 cells, 4,320 numbers): {100*full:.0f}% of the gap")
    res = {}
    for nm, f in factors.items():
        n = int(f.max().item()) + 1; r = cellfit(v, E, f, n, tr, te); res[nm] = r
        print(f"{nm:30s} ({n:4d} cells): {100*r:5.1f}% of the gap")
    # additive combinations of two single factors: fit E on v (x) one-hot(f1) and v (x) one-hot(f2)
    def additive(fs):
        F = torch.cat([v[:, :, None] * torch.nn.functional.one_hot(f, int(f.max().item()) + 1).to(dt)[:, None, :] for f in fs], 2).reshape(len(X), -1)
        A = F[tr].T @ F[tr]; Wt = torch.linalg.solve(A + 1e-6 * torch.eye(len(A), device=DEV, dtype=dt), F[tr].T @ E[tr]); P = F[te] @ Wt
        return 1 - ((P - E[te]) ** 2).sum() / ((E[te] - E[te].mean(0)) ** 2).sum()
    for combo in (("third's position", "second's position", "largest's position"), ("third's position", "second's position", "largest's position", "order"),
                  ("third's position x order", "second's position x order", "largest's position x order")):
        print(f"additive {' + '.join(combo)}: {100*additive([factors[c] for c in combo]):.1f}% of the gap")
