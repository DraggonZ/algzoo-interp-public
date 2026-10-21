"""v19's systematic gap to ARC by arrival order: E = ARC's minus v19's (D_T, D_L) fitted as a x_L + b x_S + c x_T per order,
in votes per unit. Also ARC's and v19's own coefficients per order, so the gap can be read as 'ARC does X where v19
does Y'. Only lists with the second-largest positive (where votes exist), without suppression terms."""
import sys, numpy as np, torch
sys.path.insert(0, "../../family"); sys.path.insert(0, "..")
from common import *
from algzoo.handbuilt_v19 import build_v19
from shuffle_gap import vote_unit
def fitc(v, y, c, n):
    A = torch.zeros(n, 3, 3, device=DEV, dtype=dt); B = torch.zeros(n, 3, 2, device=DEV, dtype=dt)
    A.index_add_(0, c, v[:, :, None] * v[:, None, :]); B.index_add_(0, c, v[:, :, None] * y[:, None, :])
    return torch.linalg.solve(A + 1e-6 * torch.eye(3, device=DEV, dtype=dt), B)
if __name__ == "__main__":
    g = torch.Generator(device=DEV).manual_seed(204); X = torch.randn(2000000, 10, device=DEV, dtype=dt, generator=g)
    o = X.argsort(1, descending=True); L, S, Tq = o[:, 0], o[:, 1], o[:, 2]; ar = torch.arange(len(X), device=DEV)
    v = torch.stack([X[ar, L], X[ar, S], X[ar, Tq]], 1)
    def D(W):
        u = vote_unit(W, X[:200000]); Sc = scores(W, X) / u; return torch.stack([Sc[ar, Tq] - Sc[ar, S], Sc[ar, L] - Sc[ar, S]], 1)
    Da, Dv = D(ARC), D(T(build_v19(kappa=0.5).params)); ok = ~(Dv.abs() > 50).any(1) & (v[:, 1] > 0)
    order = ((L < S).long() * 4 + (L < Tq).long() * 2 + (S < Tq).long())[ok]; vv = v[ok]
    names = {7: "LST", 6: "LTS", 3: "SLT (S, L, T)", 1: "STL", 4: "TLS", 0: "TSL"}
    sa, sv = fitc(vv, Da[ok], order, 8), fitc(vv, Dv[ok], order, 8)
    print("per unit of (largest, second, third); D_T = third minus second, D_L = largest minus second")
    for k, nm in names.items():
        print(f"{nm:14s} D_T  ARC {np.round(sa[k, :, 0].cpu().numpy(), 2)}  v19 {np.round(sv[k, :, 0].cpu().numpy(), 2)} | D_L  ARC {np.round(sa[k, :, 1].cpu().numpy(), 2)}  v19 {np.round(sv[k, :, 1].cpu().numpy(), 2)}")
