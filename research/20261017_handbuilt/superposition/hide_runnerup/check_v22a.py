"""Smoke check of v22a: on random lists, D_L in votes against the network's own rule -margin(x_L, x_S) (exact unless the
runner-up arrives at step 9 or 10, or at a near tie), for the linear first design and the final ARC-shaped margin."""
import sys, numpy as np, torch
sys.path.insert(0, "../../family")
from common import *
from build_v22a import build_v22a, LINEAR_SHAPE, ARC_SHAPE
g = torch.Generator(device=DEV).manual_seed(7); X = torch.randn(1000000, 10, device=DEV, dtype=dt, generator=g)
o = X.argsort(1, descending=True); L, S, Tq = o[:, 0], o[:, 1], o[:, 2]; ar = torch.arange(len(X), device=DEV)
xl, xs = X[ar, L], X[ar, S]
def margin(sh, l, s): return sh["a"] * s + sh["k1"] * (l - s) - sum(dk * torch.relu(l - (1 + r) * s) for r, dk in sh["knots"])
# position of the runner-up's arrival relative to the largest: did a runner-up rise happen at step 9 or 10 after L?
for nm, sh in (("v22a linear", LINEAR_SHAPE), ("v22a ARC shape", ARC_SHAPE)):
    W = T(build_v22a(sh).params); Sc = scores(W, X) / 0.25; acc = (Sc.argmax(1) == S).double().mean().item()
    err = (Sc[ar, L] - Sc[ar, S]) + margin(sh, xl, xs)
    print(f"{nm}: accuracy {100*acc:.2f}%")
    for cond, m in (("S>0, S at 0-7", (xs > 0) & (S < 8)), ("S>0, S at 8, L<8", (xs > 0) & (S == 8) & (L < 8)), ("S>0, S at 9, L<9", (xs > 0) & (S == 9) & (L != 9)), ("S>0, L at 8", (xs > 0) & (L == 8)), ("S>0, L at 9", (xs > 0) & (L == 9))):
        e = err[m].abs(); print(f"   {cond:18s} |D_L + margin| median {e.median():.4f}, 99th pct {e.quantile(0.99):.3f}, share > 0.01 {100*(e>0.01).double().mean():.2f}%")
