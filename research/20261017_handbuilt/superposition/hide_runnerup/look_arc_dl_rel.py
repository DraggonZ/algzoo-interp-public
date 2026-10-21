"""ARC's D_L / x_S (votes per unit of the second) against the relative jump r = x_L / x_S - 1, for S before L and S
after L (largest not at 9), split by the size of the second to see whether the shape is scale-free."""
import sys, numpy as np, torch
sys.path.insert(0, "../../family"); sys.path.insert(0, "../errmodel"); sys.path.insert(0, "..")
from common import *
from shuffle_gap import vote_unit
g = torch.Generator(device=DEV).manual_seed(5); X = torch.randn(4000000, 10, device=DEV, dtype=dt, generator=g)
o = X.argsort(1, descending=True); L, S = o[:, 0], o[:, 1]; ar = torch.arange(len(X), device=DEV)
xl, xs = X[ar, L], X[ar, S]
u = vote_unit(ARC, X[:200000]); Sc = scores(ARC, X) / u; y = ((Sc[ar, L] - Sc[ar, S]) / xs).cpu().numpy()
xl, xs = xl.cpu().numpy(), xs.cpu().numpy(); r = xl / xs - 1; before = (S < L).cpu().numpy(); l9 = (L == 9).cpu().numpy()
rb = [0, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.75, 1, 1.5, 2, 3, 5]
print("columns: r = x_L/x_S - 1 bins", rb)
for nm, m0 in (("S before L", before & ~l9), ("S after L", ~before & ~l9)):
    for lo, hi in ((0.2, 0.6), (0.6, 1.0), (1.0, 1.5), (1.5, 3)):
        row = []
        for j in range(len(rb) - 1):
            m = m0 & (xs >= lo) & (xs < hi) & (r >= rb[j]) & (r < rb[j + 1])
            row.append(f"{y[m].mean():6.3f}" if m.sum() > 300 else "     .")
        print(f"{nm:10s} x_S {lo}-{hi}: " + " ".join(row))
