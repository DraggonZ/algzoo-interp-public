"""Systematic gap to ARC for v19 kappa 0.5 and v22a, with the logic of errmodel/gap_parts.py (gap RMS and share of
the gap explained per arrangement, held out) and errmodel/gap_by_order.py (per-order coefficients of D_T and D_L on
(x_L, x_S, x_T), second positive). Lists with |D| > 50 votes in the hand-built network are left out, as there; the
'common lists' rows use the lists kept for both networks (v19 sets a record at step 9 to -1000 d, so those lists drop
out for v19 only)."""
import sys, numpy as np, torch
sys.path.insert(0, "../../family"); sys.path.insert(0, "../errmodel"); sys.path.insert(0, "..")
from common import *
from algzoo.handbuilt_v19 import build_v19
from shuffle_gap import vote_unit
from gap_parts import cellfit
from gap_by_order import fitc
from build_v22a import build_v22a, LINEAR_SHAPE
NETS = {"v19 kappa 0.5": T(build_v19(kappa=0.5).params), "v22a linear": T(build_v22a(LINEAR_SHAPE).params), "v22a": T(build_v22a().params)}
def Dq(W, X, ar, S, Tq, L):
    u = vote_unit(W, X[:200000]); Sc = scores(W, X) / u; return torch.stack([Sc[ar, Tq] - Sc[ar, S], Sc[ar, L] - Sc[ar, S]], 1)
if __name__ == "__main__":
    # gap_parts logic
    g = torch.Generator(device=DEV).manual_seed(203); X = torch.randn(2000000, 10, device=DEV, dtype=dt, generator=g)
    o = X.argsort(1, descending=True); L, S, Tq = o[:, 0], o[:, 1], o[:, 2]; ar = torch.arange(len(X), device=DEV)
    v = torch.stack([X[ar, L], X[ar, S], X[ar, Tq]], 1); xs = v[:, 1]
    rb = torch.bucketize(v[:, 0] / xs.abs().clamp(min=1e-3), torch.tensor([1.1, 1.25, 1.5, 2.0, 3.0], device=DEV, dtype=dt))
    c1 = L * 100 + S * 10 + Tq; c2 = c1 * 6 + rb; half = len(X) // 2
    Da = Dq(ARC, X, ar, S, Tq, L); Dc = {k: Dq(W, X, ar, S, Tq, L) for k, W in NETS.items()}
    okc = {k: ~(D.abs() > 50).any(1) for k, D in Dc.items()}; common = okc["v19 kappa 0.5"] & okc["v22a"] & okc["v22a linear"]
    print("== gap parts (as errmodel/gap_parts.py): gap = ARC's (D_T, D_L) minus the network's, votes")
    for nm in NETS:
        for lab, ok in (("own lists", okc[nm]), ("common lists", common)):
            E = Da - Dc[nm]; idx = torch.nonzero(ok)[:, 0]; tr, te = idx[idx < half], idx[idx >= half]
            r1 = cellfit(v, E, c1, 1000, tr, te); r2 = cellfit(v, E, c2, 6000, tr, te)
            print(f"{nm:14s} {lab:13s} ({100*ok.double().mean():.1f}% of lists) gap RMS {E[ok].pow(2).mean().sqrt():.3f} (D_T {E[ok,0].pow(2).mean().sqrt():.3f}, D_L {E[ok,1].pow(2).mean().sqrt():.3f}); ARC's D RMS {Da[ok].pow(2).mean().sqrt():.2f}; explained per arrangement {100*r1:.0f}%, + ratio bins {100*r2:.0f}%", flush=True)
    # gap_by_order logic
    g = torch.Generator(device=DEV).manual_seed(204); X = torch.randn(2000000, 10, device=DEV, dtype=dt, generator=g)
    o = X.argsort(1, descending=True); L, S, Tq = o[:, 0], o[:, 1], o[:, 2]; ar = torch.arange(len(X), device=DEV)
    v = torch.stack([X[ar, L], X[ar, S], X[ar, Tq]], 1)
    Da = Dq(ARC, X, ar, S, Tq, L); Dc = {k: Dq(W, X, ar, S, Tq, L) for k, W in NETS.items()}
    names = {7: "LST", 6: "LTS", 3: "SLT", 1: "STL", 4: "TLS", 0: "TSL"}
    order = (L < S).long() * 4 + (L < Tq).long() * 2 + (S < Tq).long()
    print("\n== per order (as errmodel/gap_by_order.py), coefficients per unit of (largest, second, third); second positive")
    for lab, extra in (("all lists kept by each network (v19 drops a largest at 9)", None), ("largest not at 9 (as errmodel/gap_by_order.log)", L != 9)):
        print(f"-- {lab}")
        fits = {}
        for nm, D in [("ARC", Da)] + list(Dc.items()):
            ok = (v[:, 1] > 0) & ~(Dc["v22a" if nm == "ARC" else nm].abs() > 50).any(1)   # ARC: the lists v22a keeps (all)
            if extra is not None: ok &= extra
            fits[nm] = fitc(v[ok], D[ok], order[ok], 8)
        for k, on in names.items():
            f = lambda nm, i: np.round(fits[nm][k, :, i].cpu().numpy(), 2)
            print(f"{on}  D_L: ARC {f('ARC', 1)}  v19 {f('v19 kappa 0.5', 1)}  v22a linear {f('v22a linear', 1)}  v22a {f('v22a', 1)} | D_T: ARC {f('ARC', 0)}  v19 {f('v19 kappa 0.5', 0)}  v22a {f('v22a', 0)}")
