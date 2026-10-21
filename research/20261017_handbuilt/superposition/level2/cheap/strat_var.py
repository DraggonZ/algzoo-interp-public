"""Best case for a tie-line design that enumerates arrangements and puts the third and largest on deterministic nodes:
within-stratum variance of the per-line wrong share (both ends, no double count), strata = arrangement (720) x bins
of T and L. Cost per line in list-equivalents with the prefix up to the second's position shared by both ends."""
import sys, numpy as np, torch
sys.path.insert(0, "../../../family")
from common import ARC, trained
from linelib import *
Phi = lambda x: torch.special.ndtr(x)
for name, W in (("ARC", ARC), ("114", trained(114)), ("203", trained(203))):
    n = 1_000_000; g = torch.Generator(device=DEV).manual_seed(5)
    X = torch.randn(n, 10, device=DEV, dtype=dt, generator=g)
    v, o = X.sort(1, descending=True); L, S, T = v[:, 0], v[:, 1], v[:, 2]; pS, pL, pT = o[:, 1], o[:, 0], o[:, 2]; ar = torch.arange(n, device=DEV)
    X0 = X.clone(); X0[ar, pS] = T; w, n1, f1 = first_win(W, X0, pS, pS, L - T, +1.0, max_pieces=40)
    never = w >= (L - T) * (1 - 1e-12)
    X1 = X.clone(); X1[ar, pS] = L; w2, n2, f2 = first_win(W, X1, pS, pS, L - T, -1.0, max_pieces=40)
    tot = Phi(L) - Phi(T)
    q = (((Phi(T + w) - Phi(T)) + torch.where(never, torch.zeros_like(w), Phi(L) - Phi(L - w2))) / tot).clamp(0, 1)
    cost = (pS.to(dt) * F_STEP + 2 * (10 - pS.to(dt)) * F_STEP + 2 * F_OUT).mean().item() + (f1 + f2) / n
    cost /= F_LIST
    arr = (pL * 100 + pS * 10 + pT)
    vb = q.mean().item() * (1 - q.mean().item())
    for nb in (1, 3, 6):
        tb = torch.bucketize(T, torch.quantile(T[:200000], torch.linspace(0, 1, nb + 1, device=DEV, dtype=dt)[1:-1]))
        lb = torch.bucketize(L, torch.quantile(L[:200000], torch.linspace(0, 1, nb + 1, device=DEV, dtype=dt)[1:-1]))
        key = (arr * nb + tb) * nb + lb; K = int(key.max()) + 1
        cnt = torch.zeros(K, device=DEV, dtype=dt).index_add_(0, key, torch.ones_like(q)); sm = torch.zeros(K, device=DEV, dtype=dt).index_add_(0, key, q)
        within = ((q - (sm / cnt.clamp_min(1))[key]) ** 2).mean().item()
        print(f"{name} bins {nb}x{nb}: per-line variance total {q.var().item():.5f}, within strata {within:.5f} (Bernoulli {vb:.5f}); cost {cost:.2f} lists/line; "
              f"efficiency vs sampling {vb/within/cost:.2f}x", flush=True)
