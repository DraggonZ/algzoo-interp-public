"""P8: per-order coefficients of D_L = score(largest) - score(second), in votes, on (largest, second, third), as in
hide_runnerup/eval_gap.py (largest not at 9, second positive, |D| <= 50 votes)."""
import sys; sys.path.insert(0, "/data/output/research/algzoo/20261017_handbuilt/family")
sys.path.insert(0, "/data/output/research/algzoo/20261017_handbuilt/superposition/errmodel"); sys.path.insert(0, "/data/output/research/algzoo/20261017_handbuilt/superposition")
import numpy as np, torch
from common import DEV, dt, T, ARC, scores
from shuffle_gap import vote_unit
from gap_by_order import fitc
from algzoo.handbuilt_v19 import build_v19
from algzoo.handbuilt_v23_from_v19 import build_v23f

if __name__ == "__main__":
    g = torch.Generator(device=DEV).manual_seed(204); X = torch.randn(2000000, 10, device=DEV, dtype=dt, generator=g)
    o = X.argsort(1, descending=True); L, S, Tq = o[:, 0], o[:, 1], o[:, 2]; ar = torch.arange(len(X), device=DEV)
    v = torch.stack([X[ar, L], X[ar, S], X[ar, Tq]], 1)
    nets = {"ARC": ARC, "v19": T(build_v19(kappa=0.5).params), "v23f": T(build_v23f().params), "v23f ring 0.3": T(build_v23f(ep=0.3).params)}
    order = (L < S).long() * 4 + (L < Tq).long() * 2 + (S < Tq).long()
    names = {7: "L, S, T", 6: "L, T, S", 3: "S, L, T", 1: "S, T, L", 4: "T, L, S", 0: "T, S, L"}
    fits = {}
    for nm, W in nets.items():
        u = vote_unit(W, X[:200000]); Sc = scores(W, X) / u
        D = torch.stack([Sc[ar, Tq] - Sc[ar, S], Sc[ar, L] - Sc[ar, S]], 1)
        ok = (v[:, 1] > 0) & (L != 9) & ~(D.abs() > 50).any(1)
        fits[nm] = fitc(v[ok], D[ok], order[ok], 8)
        print(f"{nm}: vote unit {u.item():.3f}, lists kept {100*ok.double().mean().item():.1f}%")
    print("arrival order | coefficient of (largest - second) on the second: " + " | ".join(nets))
    for k, on in names.items():
        print(f"{on:8s} | " + " | ".join(f"{fits[nm][k, 1, 1].item():+.2f}" for nm in nets) + "   (on the largest: " + ", ".join(f"{fits[nm][k, 0, 1].item():+.2f}" for nm in nets) + ")")
