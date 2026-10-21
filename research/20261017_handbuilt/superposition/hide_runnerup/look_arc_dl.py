"""Look at ARC's D_L (score of the largest minus score of the second, in votes) as a function of the largest and the
second, separately for S before L and S after L. Mean D_L per (x_L, x_S) bin, second positive."""
import sys, numpy as np, torch
sys.path.insert(0, "../../family"); sys.path.insert(0, "../errmodel"); sys.path.insert(0, "..")
from common import *
from shuffle_gap import vote_unit
g = torch.Generator(device=DEV).manual_seed(5); X = torch.randn(4000000, 10, device=DEV, dtype=dt, generator=g)
o = X.argsort(1, descending=True); L, S = o[:, 0], o[:, 1]; ar = torch.arange(len(X), device=DEV)
xl, xs = X[ar, L], X[ar, S]
u = vote_unit(ARC, X[:200000]); Sc = scores(ARC, X) / u; dl = (Sc[ar, L] - Sc[ar, S]).cpu().numpy()
xl, xs = xl.cpu().numpy(), xs.cpu().numpy(); before = (S < L).cpu().numpy(); l9 = (L == 9).cpu().numpy()
eb = np.array([0, .25, .5, .75, 1, 1.25, 1.5, 2, 3]); lb = [0, 0.05, 0.1, 0.2, 0.4, 0.7, 1.0, 1.5, 3]
for nm, m0 in (("S before L, L<9", before & ~l9), ("S after L", ~before)):
    print(f"\n{nm}: mean D_L (votes); rows x_S bins, cols x_L - x_S bins {lb}")
    for i in range(len(eb) - 1):
        r = []
        for j in range(len(lb) - 1):
            m = m0 & (xs >= eb[i]) & (xs < eb[i + 1]) & (xl - xs >= lb[j]) & (xl - xs < lb[j + 1])
            r.append(f"{dl[m].mean():6.2f}" if m.sum() > 200 else "     .")
        print(f"x_S {eb[i]:.2f}-{eb[i+1]:.2f}: " + " ".join(r))
