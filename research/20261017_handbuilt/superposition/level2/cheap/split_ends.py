"""Split each line's wrong mass (second-largest moving from the third up to the largest) into: the stretch touching the
2-3 tie end, the stretch touching the 1-2 tie end, and everything else. Also by the second-largest's sign region."""
import sys, numpy as np, torch
sys.path.insert(0, "../../../family")
from common import ARC, trained
from linelib import *
Phi = lambda x: torch.special.ndtr(x)
for name, W in (("ARC", ARC), ("114", trained(114)), ("203", trained(203))):
    n = 400_000; g = torch.Generator(device=DEV).manual_seed(1)
    X = torch.randn(n, 10, device=DEV, dtype=dt, generator=g)
    v, o = X.sort(1, descending=True); L, S, T = v[:, 0], v[:, 1], v[:, 2]; pS = o[:, 1]; len_ = L - T
    X0 = X.clone(); X0[torch.arange(n, device=DEV), pS] = T
    (r, lo, hi), _, _ = walk(W, X0, pS, pS, len_)
    # right stretches -> first right start and last right end per line
    first = torch.full((n,), float("inf"), device=DEV, dtype=dt).scatter_reduce(0, r, lo, "amin")
    last = torch.full((n,), -float("inf"), device=DEV, dtype=dt).scatter_reduce(0, r, hi, "amax")
    rightm = torch.zeros(n, device=DEV, dtype=dt).index_add_(0, r, Phi(T[r] + hi) - Phi(T[r] + lo))
    none = ~torch.isfinite(first)
    tot = Phi(L) - Phi(T)
    w23 = torch.where(none, tot, Phi(T + first) - Phi(T))
    w12 = torch.where(none, torch.zeros_like(tot), Phi(L) - Phi(T + last))
    wmid = (tot - rightm - w23 - w12).clamp_min(0)
    E = (tot - rightm) / tot
    print(f"{name}: error {100*E.mean():.3f}%; share at 2-3 end {(w23/tot).mean()/E.mean():.2f}, at 1-2 end {(w12/tot).mean()/E.mean():.2f}, middle {(wmid/tot).mean()/E.mean():.2f}; lines never right {100*none.double().mean():.2f}% (share {(none*E).mean()/E.mean():.2f})")
    for lo_, hi_ in ((-9, 0), (0, 0.3), (0.3, 9)):
        m = (S >= lo_) & (S < hi_)
        print(f"   S in [{lo_},{hi_}): {100*m.double().mean():5.1f}% of lists; error share 2-3 {((w23/tot)[m]).sum()/E.sum():.3f} 1-2 {((w12/tot)[m]).sum()/E.sum():.3f} mid {((wmid/tot)[m]).sum()/E.sum():.3f}")
