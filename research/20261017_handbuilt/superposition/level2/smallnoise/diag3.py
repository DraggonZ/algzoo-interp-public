"""Which competitors does the network actually lose to, given the three largest, and does a Gaussian for only the two
leads that matter (over the largest and over the third) fix the oracle's low bias? Sampling for diagnosis only."""
import numpy as np, torch
from lib import *
from diag import conditions, W
from diag2 import run_one
torch.manual_seed(2)
for label, near in [("random conditions", False), ("near-tie conditions", True)]:
    o, v = conditions(300, near); pos = torch.tensor(o, device=DEV); val = torch.tensor(v, device=DEV, dtype=dt)
    A, others = lead_matrix(pos[:, 1]); n = len(o)
    tr = np.zeros(n); o9 = np.zeros(n); o2 = np.zeros(n); o2s = np.zeros(n); lostto = np.zeros(3)
    for i in range(n):
        X = run_one(pos[i], val[i]); S = run(W, X); L = S @ A[i].T; tr[i] = (L > 0).all(1).double().mean().item()
        wrong = S.argmax(1); bad = wrong != pos[i, 1]
        for c, k in enumerate([pos[i, 0], pos[i, 2]]): lostto[c] += (bad & (wrong == k)).double().sum().item() / len(X)
        lostto[2] += (bad & (wrong != pos[i, 0]) & (wrong != pos[i, 2])).double().sum().item() / len(X)
        o9[i] = orthant(L.mean(0)[None], torch.cov(L.T)[None]).item()
        j = [int((others[i] == pos[i, 0]).nonzero()), int((others[i] == pos[i, 2]).nonzero())]
        L2 = L[:, j]; m, c = L2.mean(0), torch.cov(L2.T)
        o2[i] = orthant_k(m[None], c[None]).item()
        rest = [q for q in range(9) if q not in j]; o2s[i] = ((L2 > 0).all(1).double().mean() * (L[:, rest] > 0).all(1).double().mean()).item()
    print(f"{label}: truth {tr.mean():.4f}; mistakes per list naming the largest {lostto[0]/n:.4f}, the third {lostto[1]/n:.4f}, a smaller number {lostto[2]/n:.4f}")
    for k, r in [("oracle Gaussian, nine leads", o9), ("oracle Gaussian, two leads only", o2)]:
        print(f"  {k:34s} mean {r.mean():.4f} bias {r.mean()-tr.mean():+.4f} mean abs error {np.abs(r-tr).mean():.4f}")
    print(f"  empirical: P(two leads ok) x P(other seven ok) mean {o2s.mean():.4f} (independence check)")
