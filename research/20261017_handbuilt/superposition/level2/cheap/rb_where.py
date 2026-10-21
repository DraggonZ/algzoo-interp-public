"""Which lines carry the wrong mass: split each line's wrong mass by where on the line it sits (relative position
q = (S - T)/(L - T): near the third, middle, near the largest) and by the third's value T."""
import sys, numpy as np, torch
sys.path.insert(0, "../../../family")
from common import ARC, trained
from linelib import *
Phi = lambda x: torch.special.ndtr(x)
for name, W in (("ARC", ARC), ("114", trained(114))):
    n = 400_000; g = torch.Generator(device=DEV).manual_seed(1)
    X = torch.randn(n, 10, device=DEV, dtype=dt, generator=g)
    v, o = X.sort(1, descending=True); L, S, T = v[:, 0], v[:, 1], v[:, 2]; pS = o[:, 1]
    X0 = X.clone(); X0[torch.arange(n, device=DEV), pS] = T
    (r, lo, hi), _, _ = walk(W, X0, pS, pS, L - T)
    # wrong stretches = complement; compute right mass per stretch, then wrong via bins of the line
    edges = torch.linspace(0, 1, 41, device=DEV, dtype=dt)
    Z = (Phi(L) - Phi(T))
    rightbin = torch.zeros(n, 40, device=DEV, dtype=dt)
    for b in range(40):
        a0 = T[r] + edges[b] * (L - T)[r]; a1 = T[r] + edges[b + 1] * (L - T)[r]
        l_ = torch.maximum(T[r] + lo, a0); u_ = torch.minimum(T[r] + hi, a1)
        rightbin[:, b].index_add_(0, r, (Phi(u_) - Phi(l_)).clamp_min(0))
    tot = torch.stack([Phi(T + edges[b + 1] * (L - T)) - Phi(T + edges[b] * (L - T)) for b in range(40)], 1)
    wrongbin = ((tot - rightbin) / Z[:, None]).clamp_min(0)
    w = wrongbin.sum(1)
    print(f"{name}: error {100*w.mean():.3f}%; by line part (share of error): ", end="")
    sh = wrongbin.mean(0) / w.mean()
    print("first 1/40 %.2f, 1/40-1/4 %.2f, 1/4-3/4 %.2f, 3/4-39/40 %.2f, last 1/40 %.2f" % (sh[0], sh[1:10].sum(), sh[10:30].sum(), sh[30:39].sum(), sh[39]))
    for lo_, hi_ in ((-9, -0.5), (-0.5, 0), (0, 0.5), (0.5, 1), (1, 9)):
        m = (T >= lo_) & (T < hi_)
        print(f"   T in [{lo_},{hi_}): {100*m.double().mean():5.1f}% of lines, {100*w[m].sum()/w.sum():5.1f}% of error, {100*(w[m]**2).sum()/(w**2).sum():5.1f}% of sum of squares")
