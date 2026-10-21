"""shuffle_gap.py with v21 (28 neurons) and v19 (kappa 0.5) as the clean part: ARC = clean + gap on each list (scores in
votes, centred); the gap borrowed from another list with the same arrangement and ratio bin. Does 'hand-built clean
network + borrowed gap' predict ARC's accuracy and near-tie behaviour?"""
import sys, numpy as np, torch
sys.path.insert(0, "../../family"); sys.path.insert(0, "..")
from common import *
from algzoo.handbuilt_v21 import build_v21
from algzoo.handbuilt_v19 import build_v19
import shuffle_gap as sg
if __name__ == "__main__":
    g = torch.Generator(device=DEV).manual_seed(201); X = torch.randn(1000000, 10, device=DEV, dtype=dt, generator=g)
    o = X.argsort(1, descending=True); L, S, Tq = o[:, 0], o[:, 1], o[:, 2]; ar = torch.arange(len(X), device=DEV)
    xs = X[ar, S]; ratio = X[ar, L] / xs.abs().clamp(min=1e-3); rb = torch.bucketize(ratio, torch.tensor([1.1, 1.25, 1.5, 2.0, 3.0], device=DEV, dtype=dt))
    cell = (L * 100 + S * 10 + Tq) * 6 + rb; near = (xs > 0) & ((xs - X[ar, Tq]) / xs < 0.05)
    ua = sg.vote_unit(ARC, X); Sa = scores(ARC, X) / ua; Sa = Sa - Sa.mean(1, keepdim=True); aA = Sa.argmax(1); wrong = aA != S
    def borrow(Sc, E, seed):
        gg = torch.Generator(device=DEV).manual_seed(seed)
        order = torch.argsort(cell.double() + torch.rand(len(X), device=DEV, dtype=dt, generator=gg) * 0.5); cs = cell[order]
        st, en = torch.searchsorted(cs, cs), torch.searchsorted(cs, cs, right=True)
        pick = torch.minimum(st + (torch.rand(len(X), device=DEV, dtype=dt, generator=gg) * (en - st)).long(), en - 1)
        src = torch.empty_like(order); src[order] = order[pick]
        return Sc + E[src] * (xs.abs() / xs[src].abs().clamp(min=1e-3)).clamp(max=10)[:, None]
    def rep(nm, Sp):
        a = Sp.argmax(1); tw = (a == Tq) & near
        byr = " ".join(f"{100*(tw & (rb == b)).double().sum()/(near & (rb == b)).double().sum():4.0f}" for b in range(6))
        print(f"{nm:44s} acc {100*(a == S).double().mean():6.2f}%  ARC mistakes reproduced {100*((a == aA) & wrong).double().sum()/wrong.double().sum():5.1f}% | near ties third wins {100*tw.double().sum()/near.double().sum():5.1f}% | by ratio {byr}", flush=True)
    rep("ARC", Sa)
    for nm, W in (("v21", T(build_v21())), ("v19 kappa 0.5", T(build_v19(kappa=0.5).params))):
        uc = sg.vote_unit(W, X); Sc = scores(W, X) / uc; Sc = Sc - Sc.mean(1, keepdim=True); E = Sa - Sc
        rep(f"clean {nm}", Sc); rep(f"clean {nm} + gap borrowed within cell", borrow(Sc, E, 0))
        print(f"   gap size: RMS {E.pow(2).mean().sqrt():.2f} votes per unit-ish; ARC scores RMS {Sa.pow(2).mean().sqrt():.2f}")
