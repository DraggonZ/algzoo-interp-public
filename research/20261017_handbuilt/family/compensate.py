"""Do the memory and the cancelling neurons compensate each other in the final decision? At near ties (third within 5%
of the second, second > 0): each group's contribution to D = score_S - score_T (its final activity times its readout
weights), per unit of x_S. Spread of each, correlation between them, and spread of their sum, for ARC and two other
trained networks (groups found as in ARC: largest-input-weight-negative neurons... here ARC only uses its labels; for
other networks all neurons are split by their own readout contribution sign, so only ARC is reported by group)."""
import numpy as np, torch
from common import *
if __name__ == "__main__":
    X = near_lists(200000, seed=3); L, S, Tq = top3(X); ar = torch.arange(len(X), device=DEV); xs = X[ar, S]
    wi, wh, wo = ARC
    h = torch.zeros(len(X), 16, device=DEV, dtype=dt)
    for t in range(10): h = torch.relu(h @ wh.T + X[:, t:t + 1] * wi)
    G = {"cancelling (1,6,7,8)": [1, 6, 7, 8], "memory (0,3,10-15)": [0, 3, 10, 11, 12, 13, 14, 15], "store+detector (2,4)": [2, 4], "gates (5,9)": [5, 9]}
    C = {k: ((h[:, idx] * (wo[S][:, idx] - wo[Tq][:, idx])).sum(1) / xs) for k, idx in G.items()}
    D = sum(C.values())
    print(f"ARC near ties: D/x_S median {D.median():.2f}, spread (10-90%) [{D.quantile(.1):.1f}, {D.quantile(.9):.1f}], wrong {100*(D<0).double().mean():.1f}%")
    for k, v in C.items(): print(f"  {k:22s} median {v.median():+8.2f}  spread [{v.quantile(.1):+8.1f}, {v.quantile(.9):+8.1f}]  SD {v.std():7.1f}")
    ks = list(C); M = torch.stack([C[k] for k in ks]); cc = torch.corrcoef(M)
    print("  correlations: " + "; ".join(f"{ks[i].split()[0]}-{ks[j].split()[0]} {cc[i,j]:+.2f}" for i in range(4) for j in range(i + 1, 4)))
    cm = C[ks[0]] + C[ks[1]]; print(f"  cancelling + memory: SD {cm.std():.1f} (each {C[ks[0]].std():.1f}, {C[ks[1]].std():.1f}); total D SD {D.std():.1f}")
