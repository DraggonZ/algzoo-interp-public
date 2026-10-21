"""Where the variance of the line estimator sits: random lists, wrong mass touching each end of the line (no double
count), normalized per line (Rao-Blackwell). Shares of error and of sum of squares by the size of the wrong stretch
relative to the line, by whether the line is never right, and by the largest's position."""
import sys, numpy as np, torch
sys.path.insert(0, "../../../family")
from common import ARC, trained
from linelib import *
Phi = lambda x: torch.special.ndtr(x)
for name, W in (("ARC", ARC), ("114", trained(114))):
    n = 400_000; g = torch.Generator(device=DEV).manual_seed(1)
    X = torch.randn(n, 10, device=DEV, dtype=dt, generator=g)
    v, o = X.sort(1, descending=True); L, S, T = v[:, 0], v[:, 1], v[:, 2]; pS = o[:, 1]; pL = o[:, 0]; ar = torch.arange(n, device=DEV)
    X0 = X.clone(); X0[ar, pS] = T; w, _, _ = first_win(W, X0, pS, pS, L - T, +1.0, max_pieces=40)
    never = w >= (L - T) * (1 - 1e-12)
    X1 = X.clone(); X1[ar, pS] = L; w2, _, _ = first_win(W, X1, pS, pS, L - T, -1.0, max_pieces=40)
    tot = Phi(L) - Phi(T)
    q = ((Phi(T + w) - Phi(T)) + torch.where(never, torch.zeros_like(w), Phi(L) - Phi(L - w2))) / tot
    q = q.clamp(0, 1).cpu().numpy(); never = never.cpu().numpy(); pLn = pL.cpu().numpy(); Tn = T.cpu().numpy()
    print(f"{name}: error {100*q.mean():.3f}%  var {q.var():.5f} (Bernoulli {q.mean()*(1-q.mean()):.5f})")
    for a, b in ((0, 0.05), (0.05, 0.2), (0.2, 0.5), (0.5, 0.999), (0.999, 1.01)):
        m = (q > a) & (q <= b); print(f"   line wrong share in ({a},{b}]: {100*m.mean():5.2f}% of lines, {100*q[m].sum()/q.sum():5.1f}% of error, {100*(q[m]**2).sum()/(q**2).sum():5.1f}% of sum of squares")
    print("   by largest's position, error share: " + " ".join(f"{p}:{q[pLn==p].sum()/q.sum():.2f}" for p in range(10)))
    print("   never-right lines by T: " + " ".join(f"T<{b}:{np.mean(never & (Tn<b)) / max(np.mean(never),1e-9):.2f}" for b in (0, 0.5, 1)))
