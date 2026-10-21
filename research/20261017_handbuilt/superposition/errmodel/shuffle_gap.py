"""Clean network plus an error model. Clean: v7z (v7 + z + ninth slot, 18 neurons, exact role copy, 98.3%). Gap on each
list: e = ARC's scores / ARC's vote per unit - v7z's scores / v7z's vote per unit, centred over positions (exact by
construction: ARC = clean + e). Error model: the gap treated as noise that knows only the arrangement of the three
largest (720 cells) and the largest / second ratio (6 bins): each list gets the gap of another random list from the same
cell, rescaled by the ratio of their second-largest values. Predicted scores = clean + borrowed gap. If this predicts
ARC's accuracy and near-tie behaviour, the description 'clean network + gaps as noise' holds at this level. Controls:
the gap borrowed from a random list of any cell; the gap scaled by 0.5 and 1.5. Scored on random lists (not the grid:
the borrowed gaps come from random lists) by accuracy, near-tie third wins by ratio and order, and agreement with ARC
on ARC's mistakes (ARC's mistakes reproduced, random lists)."""
import sys, numpy as np, torch
sys.path.insert(0, "../../family"); sys.path.insert(0, "..")
from common import *
from baseline_v7z import nets
import score
def vote_unit(W, X):
    J = jac(W, X[:20000]); o = X[:20000].argsort(1, descending=True); L = o[:, 0]
    kept = (X[:20000] > 0) & (torch.arange(10, device=DEV)[None] != L[:, None]); return J[:, torch.arange(10), torch.arange(10)][kept].median()
if __name__ == "__main__":
    g = torch.Generator(device=DEV).manual_seed(201); X = torch.randn(1000000, 10, device=DEV, dtype=dt, generator=g)
    Wc = nets()["v7 + z + ninth slot (18)"]; ua, uc = vote_unit(ARC, X), vote_unit(Wc, X)
    Sa = scores(ARC, X) / ua; Sc = scores(Wc, X) / uc; Sa = Sa - Sa.mean(1, keepdim=True); Sc = Sc - Sc.mean(1, keepdim=True); E = Sa - Sc
    o = X.argsort(1, descending=True); L, S, Tq = o[:, 0], o[:, 1], o[:, 2]; ar = torch.arange(len(X), device=DEV)
    xs = X[ar, S]; ratio = X[ar, L] / xs.abs().clamp(min=1e-3)
    cell = (L * 100 + S * 10 + Tq) * 6 + torch.bucketize(ratio, torch.tensor([1.1, 1.25, 1.5, 2.0, 3.0], device=DEV, dtype=dt))
    def borrow(within_cell, k=1.0, seed=0):
        gg = torch.Generator(device=DEV).manual_seed(seed)
        if within_cell:
            order = torch.argsort(cell * 1.0 + torch.rand(len(X), device=DEV, dtype=dt, generator=gg) * 0.5)
            cs = cell[order]; start = torch.searchsorted(cs, cs, right=False); end = torch.searchsorted(cs, cs, right=True)
            pick = start + (torch.rand(len(X), device=DEV, dtype=dt, generator=gg) * (end - start)).long()
            pick = torch.minimum(pick, end - 1); src = torch.empty_like(order); src[order] = order[pick]
        else:
            src = torch.randint(0, len(X), (len(X),), device=DEV, generator=gg)
        # map the borrowed gap from the donor's positions to this list's: same cell => same positions of L, S, T
        scale = (xs.abs() / X[src, S[src]].abs().clamp(min=1e-3)).clamp(max=10)
        return Sc + k * E[src] * scale[:, None]
    y = S; aARC = Sa.argmax(1); wrong = aARC != y
    near = (X[ar, S] > 0) & ((X[ar, S] - X[ar, Tq]) / X[ar, S] < 0.05)
    rb = torch.bucketize(ratio, torch.tensor([1.1, 1.25, 1.5, 2.0, 3.0], device=DEV, dtype=dt))
    ordc = (L < S).long() * 4 + (L < Tq).long() * 2 + (S < Tq).long()
    def report(nm, Sp):
        a = Sp.argmax(1); tw = (a == Tq) & near
        byr = " ".join(f"{100*tw[rb == b].double().sum()/near[rb == b].double().sum():4.0f}" for b in range(6))
        byo = []
        for onm in ["LST", "LTS", "SLT", "TLS", "STL", "TSL"]:
            Lpos, Spos, Tpos = (onm.index(c) for c in "LST"); m = near & ((L < S) == (Lpos < Spos)) & ((L < Tq) == (Lpos < Tpos)) & ((S < Tq) == (Spos < Tpos))
            byo.append(f"{100*tw[m].double().sum()/m.double().sum():4.0f}")
        print(f"{nm:44s} acc {100*(a == y).double().mean():6.2f}%  ARC mistakes reproduced {100*((a == aARC) & wrong).double().sum()/wrong.double().sum():5.1f}% | near ties third wins {100*tw.double().sum()/near.double().sum():5.1f}% | ratio {byr} | order {' '.join(byo)}")
    report("ARC", Sa); report("clean (v7z)", Sc)
    for nm, (wc, k) in {"clean + gap borrowed within cell": (True, 1.0), "clean + gap borrowed within cell x0.5": (True, 0.5),
                        "clean + gap borrowed within cell x1.5": (True, 1.5), "clean + gap borrowed from any list": (False, 1.0)}.items():
        report(nm, borrow(wc, k))
    for s in (1, 2): report(f"clean + gap within cell, other draw {s}", borrow(True, 1.0, seed=s))
