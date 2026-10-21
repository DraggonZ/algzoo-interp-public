"""Rao-Blackwell check: for random lists, move the second-largest from the third up to the largest with the rest fixed;
P(right | rest) = density-weighted share of that line where the network names it. Mean should equal accuracy; its
variance against the Bernoulli variance says how much the line integral alone saves."""
import sys, time, numpy as np, torch
sys.path.insert(0, "../../../family")
from common import ARC, trained, scores
from linelib import *
Phi = lambda x: torch.special.ndtr(x)
def rb(W, n, seed):
    g = torch.Generator(device=DEV).manual_seed(seed)
    X = torch.randn(n, 10, device=DEV, dtype=dt, generator=g)
    v, o = X.sort(1, descending=True); L, S, T = v[:, 0], v[:, 1], v[:, 2]; pS = o[:, 1]
    X0 = X.clone(); X0[torch.arange(n, device=DEV), pS] = T
    (r, lo, hi), pieces, fl = walk(W, X0, pS, pS, L - T)
    mass = torch.zeros(n, device=DEV, dtype=dt).index_add_(0, r, Phi(T[r] + hi) - Phi(T[r] + lo))
    p = mass / (Phi(L) - Phi(T))
    right = (scores(W, X).argmax(1) == pS).to(dt)
    return p, right, pieces / n, fl / n
for name, W in (("ARC", ARC), ("114", trained(114)), ("203", trained(203))):
    t0 = time.time(); p, right, pc, fl = rb(W, 400_000, 1)
    print(f"{name}: RB mean {100*p.mean():.3f} (SE {100*p.std()/np.sqrt(len(p)):.3f})  direct {100*right.mean():.3f}  "
          f"var ratio {p.var().item()/right.var().item():.4f}  pieces/line {pc:.2f}  ops/line {fl:.0f} = {fl/F_LIST:.2f} lists  "
          f"efficiency {right.var().item()/p.var().item()/(fl/F_LIST):.1f}x  {time.time()-t0:.0f}s", flush=True)
