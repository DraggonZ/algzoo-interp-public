"""Tie-surface estimator, randomized version, to measure its variance and cost.
2-3 end: nine iid normals at the other positions; the moving position k (uniform) starts at T (the second of the
nine) and moves up; c23 = Phi(T + w) - Phi(T), w the first a where k wins (capped at L - T). error_23 = E[10 c23].
1-2 end: k starts at L and moves down; c12 = Phi(L) - Phi(L - w'), w' capped at L - T. error_12 = E[10 c12].
Also: share of variance between the 720 position arrangements."""
import sys, numpy as np, torch
sys.path.insert(0, "../../../family")
from common import ARC, trained, scores
from linelib import *
Phi = lambda x: torch.special.ndtr(x)
tr = np.load("../../../hilton/propagate/across.npy")[:, 0]
for name, W, t in (("ARC", ARC, tr[0]), ("114", trained(114), tr[15]), ("203", trained(203), tr[20])):
    n = 400_000; g = torch.Generator(device=DEV).manual_seed(2)
    Y = torch.randn(n, 9, device=DEV, dtype=dt, generator=g); k = torch.randint(0, 10, (n,), device=DEV, generator=g)
    X = torch.zeros(n, 10, device=DEV, dtype=dt); ar = torch.arange(10, device=DEV)
    oth = torch.stack([torch.cat([ar[:j], ar[j + 1:]]) for j in range(10)])[k]
    X.scatter_(1, oth, Y)
    v, o = Y.sort(1, descending=True); L, T = v[:, 0], v[:, 1]; pL = oth.gather(1, o[:, :1])[:, 0]; pT = oth.gather(1, o[:, 1:2])[:, 0]
    X23 = X.clone(); X23[torch.arange(n, device=DEV), k] = T
    w, pc23, f23 = first_win(W, X23, k, k, L - T, +1.0)
    c23 = Phi(T + w) - Phi(T)
    X12 = X.clone(); X12[torch.arange(n, device=DEV), k] = L
    w2, pc12, f12 = first_win(W, X12, k, k, L - T, -1.0)
    c12 = Phi(L) - Phi(L - w2)
    e23, e12 = 10 * c23, 10 * c12; est = e23 + e12
    cost = (2 * F_LIST + (f23 + f12) / n) / F_LIST     # forward at each tie point + walks, in list-equivalents
    acc = 1 - est.mean().item()
    vb = t * (1 - t)
    # between-arrangement share
    arr = (k * 100 + pL * 10 + pT)
    cnt = torch.zeros(1000, device=DEV, dtype=dt).index_add_(0, arr, torch.ones_like(est)); sm = torch.zeros(1000, device=DEV, dtype=dt).index_add_(0, arr, est)
    mean_a = (sm / cnt.clamp_min(1))[arr]; within = ((est - mean_a) ** 2).mean().item()
    print(f"{name}: estimate {100*acc:.3f} (true {100*t:.3f}; missing middle), 2-3 {100*e23.mean():.3f} 1-2 {100*e12.mean():.3f};  "
          f"var per sample {est.var().item():.5f} (Bernoulli {vb:.5f}); within-arrangement {within:.5f}; pieces {pc23.mean():.2f}+{pc12.mean():.2f}; cost {cost:.2f} lists; "
          f"efficiency vs sampling {vb/est.var().item()/cost:.1f}x, stratified by arrangement {vb/within/cost:.1f}x", flush=True)
