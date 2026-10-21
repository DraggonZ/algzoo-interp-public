"""Which half of v19's gap matters? Decision quantities in each network's votes: D_T (third minus second), D_L (largest
minus second). Mixes: v19's D_T with ARC's per-order rule for D_L (a x_L + b x_S + c x_T per arrival order, 36 numbers
in all), and ARC's per-order D_T rule with v19's D_L; both per-order rules; each plus ARC's leftover from its own
per-order rules borrowed within arrangement and ratio bin. Choice among the three largest; compared with ARC."""
import sys, numpy as np, torch
sys.path.insert(0, "../../family"); sys.path.insert(0, "..")
from common import *
from algzoo.handbuilt_v19 import build_v19
from shuffle_gap import vote_unit
from gap_by_order import fitc
if __name__ == "__main__":
    def lists(seed):
        g = torch.Generator(device=DEV).manual_seed(seed); X = torch.randn(1000000, 10, device=DEV, dtype=dt, generator=g)
        o = X.argsort(1, descending=True); L, S, Tq = o[:, 0], o[:, 1], o[:, 2]; ar = torch.arange(len(X), device=DEV)
        v = torch.stack([X[ar, L], X[ar, S], X[ar, Tq]], 1); order = (L < S).long() * 4 + (L < Tq).long() * 2 + (S < Tq).long()
        return X, L, S, Tq, ar, v, order
    W19 = T(build_v19(kappa=0.5).params)
    def D(W, X, L, S, Tq, ar):
        u = vote_unit(W, X[:200000]); Sc = scores(W, X) / u; return torch.stack([Sc[ar, Tq] - Sc[ar, S], Sc[ar, L] - Sc[ar, S]], 1)
    X, L, S, Tq, ar, v, order = lists(205)                       # fitting lists
    ruleA = fitc(v, D(ARC, X, L, S, Tq, ar), order, 8)
    X, L, S, Tq, ar, v, order = lists(206)                       # test lists
    Da, Dv = D(ARC, X, L, S, Tq, ar), D(W19, X, L, S, Tq, ar); Ra = (v[:, :, None] * ruleA[order]).sum(1); Ea = Da - Ra
    xs = v[:, 1]; rb = torch.bucketize(v[:, 0] / xs.abs().clamp(min=1e-3), torch.tensor([1.1, 1.25, 1.5, 2.0, 3.0], device=DEV, dtype=dt))
    cell = (L * 100 + S * 10 + Tq) * 6 + rb; near = (xs > 0) & ((xs - v[:, 2]) / xs < 0.05)
    gg = torch.Generator(device=DEV).manual_seed(0); od = torch.argsort(cell.double() + torch.rand(len(X), device=DEV, dtype=dt, generator=gg) * 0.5); cs = cell[od]
    st, en = torch.searchsorted(cs, cs), torch.searchsorted(cs, cs, right=True); pick = torch.minimum(st + (torch.rand(len(X), device=DEV, dtype=dt, generator=gg) * (en - st)).long(), en - 1)
    src = torch.empty_like(od); src[od] = od[pick]; noise = Ea[src] * (xs.abs() / xs[src].abs().clamp(min=1e-3)).clamp(max=10)[:, None]
    def choice(Dd): return torch.where((Dd[:, 0] > 0) & (Dd[:, 0] > Dd[:, 1]), 2, torch.where((Dd[:, 1] > 0) & (Dd[:, 1] >= Dd[:, 0]), 0, 1))
    ca = choice(Da); wr = ca != 1
    def rep(nm, Dd):
        c = choice(Dd); tw = (c == 2) & near
        print(f"{nm:52s} right {100*(c == 1).double().mean():6.2f}% | ARC's wrong choices reproduced {100*((c == ca) & wr).double().sum()/wr.double().sum():5.1f}% (names largest {100*((c == ca) & (ca == 0)).double().sum()/(ca == 0).double().sum():4.1f}%, third {100*((c == ca) & (ca == 2)).double().sum()/(ca == 2).double().sum():4.1f}%) | near ties third wins {100*tw.double().sum()/near.double().sum():5.1f}%")
    rep("ARC", Da); rep("ARC's per-order rules (36 numbers)", Ra); rep("ARC's per-order rules + borrowed leftover", Ra + noise)
    rep("v19", Dv); rep("v19 + borrowed leftover", Dv + noise)
    mix1 = torch.stack([Dv[:, 0], Ra[:, 1]], 1); mix2 = torch.stack([Ra[:, 0], Dv[:, 1]], 1)
    rep("v19's third-vs-second, ARC's largest-vs-second rule", mix1); rep("  + borrowed leftover", mix1 + noise)
    rep("ARC's third-vs-second rule, v19's largest-vs-second", mix2); rep("  + borrowed leftover", mix2 + noise)
