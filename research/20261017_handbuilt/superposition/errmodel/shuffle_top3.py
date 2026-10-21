"""Clean network + borrowed gap, on the decision quantities only: D_T = score_T - score_S and D_L = score_L - score_S
(in each network's votes). Gap = ARC's D minus the clean network's D, per list. Borrowed from another list with the same
arrangement and ratio bin, rescaled by the second-largest. The choice is made among the three largest. Clean parts: v21,
v19 kappa 0.5, v7z, and ARC's own typical rule (shuffle_typical.py) for reference."""
import sys, numpy as np, torch
sys.path.insert(0, "../../family"); sys.path.insert(0, "..")
from common import *
from algzoo.handbuilt_v21 import build_v21
from algzoo.handbuilt_v19 import build_v19
from baseline_v7z import nets
from shuffle_gap import vote_unit
if __name__ == "__main__":
    g = torch.Generator(device=DEV).manual_seed(202); X = torch.randn(1000000, 10, device=DEV, dtype=dt, generator=g)
    o = X.argsort(1, descending=True); L, S, Tq = o[:, 0], o[:, 1], o[:, 2]; ar = torch.arange(len(X), device=DEV)
    xs = X[ar, S]; ratio = X[ar, L] / xs.abs().clamp(min=1e-3); rb = torch.bucketize(ratio, torch.tensor([1.1, 1.25, 1.5, 2.0, 3.0], device=DEV, dtype=dt))
    cell = (L * 100 + S * 10 + Tq) * 6 + rb; near = (xs > 0) & ((xs - X[ar, Tq]) / xs < 0.05)
    def D(W):
        u = vote_unit(W, X); Sc = scores(W, X) / u; return torch.stack([Sc[ar, Tq] - Sc[ar, S], Sc[ar, L] - Sc[ar, S]], 1)
    def choice(Dd): return torch.where((Dd[:, 0] > 0) & (Dd[:, 0] > Dd[:, 1]), 2, torch.where((Dd[:, 1] > 0) & (Dd[:, 1] >= Dd[:, 0]), 0, 1))
    Da = D(ARC); ca = choice(Da); wr = ca != 1
    def borrow(Dc, E, seed):
        gg = torch.Generator(device=DEV).manual_seed(seed)
        order = torch.argsort(cell.double() + torch.rand(len(X), device=DEV, dtype=dt, generator=gg) * 0.5); cs = cell[order]
        st, en = torch.searchsorted(cs, cs), torch.searchsorted(cs, cs, right=True)
        pick = torch.minimum(st + (torch.rand(len(X), device=DEV, dtype=dt, generator=gg) * (en - st)).long(), en - 1)
        src = torch.empty_like(order); src[order] = order[pick]
        return Dc + E[src] * (xs.abs() / xs[src].abs().clamp(min=1e-3)).clamp(max=10)[:, None]
    def rep(nm, Dd):
        c = choice(Dd); tw = (c == 2) & near
        byr = " ".join(f"{100*(tw & (rb == b)).double().sum()/(near & (rb == b)).double().sum():4.0f}" for b in range(6))
        print(f"{nm:44s} right among top three {100*(c == 1).double().mean():6.2f}% | ARC's wrong choices reproduced {100*((c == ca) & wr).double().sum()/wr.double().sum():5.1f}% | near ties third wins {100*tw.double().sum()/near.double().sum():5.1f}% | by ratio {byr}", flush=True)
    rep("ARC", Da)
    for nm, W in (("v21", T(build_v21())), ("v19 kappa 0.5", T(build_v19(kappa=0.5).params)), ("v7z", nets()["v7 + z + ninth slot (18)"])):
        Dc = D(W); E = Da - Dc
        big = (Dc.abs() > 50).any(1)
        rep(f"clean {nm}", Dc); rep(f"clean {nm} + gap borrowed within cell", borrow(Dc, E, 0))
        print(f"   gap RMS {E[~big].pow(2).mean().sqrt():.2f} votes (ARC's D RMS {Da.pow(2).mean().sqrt():.2f}); lists with a huge clean D (suppression terms): {100*big.double().mean():.2f}%")
