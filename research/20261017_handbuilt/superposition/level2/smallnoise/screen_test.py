"""Can the cheap 'two' model screen which grid points need the expensive 'fourth' model? ARC, grid 16^3, Q4=4."""
import numpy as np, torch, models, grids
from lib import *
W = load_net("ARC"); V, wt = grids.get("split-gl", 16, 16); V = torch.tensor(V, device=DEV, dtype=dt); wt = torch.tensor(wt, device=DEV, dtype=dt)
nV = len(wt); pos = TRIPLES.repeat_interleave(nV, 0); val = V.repeat(len(TRIPLES), 1); w = wt.repeat(len(TRIPLES)); w = w / w.sum()
P2, P4, Z = [], [], []
for i in range(0, len(w), 4000):
    sl = slice(i, i + 4000)
    P2.append(models.model_two(W, pos[sl], val[sl])[0]); P4.append(models.model_fourth(W, pos[sl], val[sl], 4)[0])
P2 = torch.cat(P2); P4 = torch.cat(P4)
print(f"two {100*(w*P2).sum():.3f}%  fourth {100*(w*P4).sum():.3f}%  total |difference| mass {100*(w*(P2-P4).abs()).sum():.3f} points")
for thr in [1e-2, 1e-3, 1e-4, 1e-6]:
    m = (P2 > thr) & (P2 < 1 - thr)
    hyb = torch.where(m, P4, P2)
    print(f"  fourth only where 'two' gives {thr:g} < P < 1-{thr:g}: {100*m.double().mean():.1f}% of points ({100*(w*m).sum():.1f}% of mass); hybrid {100*(w*hyb).sum():.3f}%")
