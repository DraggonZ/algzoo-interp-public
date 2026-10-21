"""Is ARC's leftover noise-like relative to its typical behaviour? Typical: one linear rule per exact arrangement of the
three largest (../../family/config_rule.py, 4,320 numbers, fitted on other random lists): the third's and the largest's
score minus the second's as a x_L + b x_S + c x_T. Leftover on each list: ARC's two differences minus the rule's
(exact). Borrow the leftover from another list in the same arrangement and ratio bin, rescaled by the second-largest;
predict the choice among the three largest. Compare accuracy and near-tie behaviour with ARC's (restricted to the three
largest, like the rule)."""
import sys, numpy as np, torch
sys.path.insert(0, "../../family")
from common import *
from config_rule import fit, cells
if __name__ == "__main__":
    sol = fit("exact arrangement")
    g = torch.Generator(device=DEV).manual_seed(202); X = torch.randn(1000000, 10, device=DEV, dtype=dt, generator=g)
    o = X.argsort(1, descending=True); L, S, Tq = o[:, 0], o[:, 1], o[:, 2]; ar = torch.arange(len(X), device=DEV)
    v = torch.stack([X[ar, L], X[ar, S], X[ar, Tq]], 1); Sa = scores(ARC, X)
    Da = torch.stack([Sa[ar, Tq] - Sa[ar, S], Sa[ar, L] - Sa[ar, S]], 1); Dr = (v[:, :, None] * sol[cells(X, "exact arrangement")]).sum(1); E = Da - Dr
    xs = v[:, 1]; ratio = v[:, 0] / xs.abs().clamp(min=1e-3); rb = torch.bucketize(ratio, torch.tensor([1.1, 1.25, 1.5, 2.0, 3.0], device=DEV, dtype=dt))
    cell = cells(X, "exact arrangement") * 6 + rb
    def borrow(seed, within=True):
        gg = torch.Generator(device=DEV).manual_seed(seed)
        if within:
            order = torch.argsort(cell.double() + torch.rand(len(X), device=DEV, dtype=dt, generator=gg) * 0.5); cs = cell[order]
            st, en = torch.searchsorted(cs, cs), torch.searchsorted(cs, cs, right=True)
            pick = torch.minimum(st + (torch.rand(len(X), device=DEV, dtype=dt, generator=gg) * (en - st)).long(), en - 1)
            src = torch.empty_like(order); src[order] = order[pick]
        else: src = torch.randint(0, len(X), (len(X),), device=DEV, generator=gg)
        return Dr + E[src] * (xs.abs() / xs[src].abs().clamp(min=1e-3)).clamp(max=10)[:, None]
    near = (xs > 0) & ((xs - v[:, 2]) / xs < 0.05)
    def rep(nm, D):
        third = (D[:, 0] > 0) & (D[:, 0] > D[:, 1]); largest = (D[:, 1] > 0) & (D[:, 1] >= D[:, 0]); right = ~third & ~largest
        choice = torch.where(third, 2, torch.where(largest, 0, 1)); ca = torch.where((Da[:, 0] > 0) & (Da[:, 0] > Da[:, 1]), 2, torch.where((Da[:, 1] > 0) & (Da[:, 1] >= Da[:, 0]), 0, 1))
        wr = ca != 1
        byr = " ".join(f"{100*(third & near & (rb == b)).double().sum()/(near & (rb == b)).double().sum():4.0f}" for b in range(6))
        print(f"{nm:40s} right among the top three {100*right.double().mean():6.2f}% | ARC's wrong choices reproduced {100*((choice == ca) & wr).double().sum()/wr.double().sum():5.1f}% | near ties third wins {100*(third & near).double().sum()/near.double().sum():5.1f}% | by ratio {byr}")
    rep("ARC (restricted to the top three)", Da); rep("typical rule alone", Dr)
    for s in range(2): rep(f"typical + leftover borrowed in cell (draw {s})", borrow(s))
    rep("typical + leftover borrowed from any list", borrow(9, False))
    print(f"leftover size: RMS of ARC's differences {Da.pow(2).mean().sqrt():.2f}, of the leftover {E.pow(2).mean().sqrt():.2f}")
