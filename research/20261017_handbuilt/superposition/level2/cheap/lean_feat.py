"""What in the seven smaller numbers moves the lean at a 2nd-3rd tie? R^2 of the lean on simple summaries:
each value (linear), the running records before the largest arrives (value and position of the max and second max
of the smaller numbers before it), and both."""
import sys, numpy as np, torch
sys.path.insert(0, "../../../family")
from common import ARC, trained, scores
from linelib import *
Phi = lambda x: torch.special.ndtr(x)
def r2(F, y):
    F = np.c_[np.ones(len(F)), F]; b = np.linalg.lstsq(F, y, rcond=None)[0]; return 1 - np.var(y - F @ b) / np.var(y)
for name, W in (("ARC", ARC), ("114", trained(114))):
    g = torch.Generator(device=DEV).manual_seed(0); rng = np.random.default_rng(3); print(name)
    for trial in range(10):
        pL, pS, pT = rng.choice(10, 3, replace=False); L, T = 1.6, 0.8
        rest = [i for i in range(10) if i not in (pL, pS, pT)]
        n = 20000; u = torch.rand(n, 7, device=DEV, dtype=dt, generator=g) * Phi(torch.tensor(T, dtype=dt)); Z = torch.special.ndtri(u)
        X = torch.zeros(n, 10, device=DEV, dtype=dt); X[:, rest] = Z; X[:, pL] = L; X[:, pS] = T; X[:, pT] = T
        s = scores(W, X); lam = (s[:, pS] - s[:, pT]).cpu().numpy(); Xn = X.cpu().numpy()
        first = min(pL, pS, pT)
        before = [i for i in rest if i < first]
        lin = Xn[:, rest]; feats = [lin, np.maximum(lin, 0)]
        # record features among smaller numbers before the first of the three largest: one-hot position of max x value
        if before:
            B = Xn[:, before]; am = B.argmax(1); mx = B.max(1)
            oh = np.eye(len(before))[am]; rec = np.c_[oh * mx[:, None], oh]
        else: rec = np.zeros((n, 1))
        print(f"  L@{pL} S@{pS} T@{pT} ({len(before)} before): sd {lam.std():.3f}  R2 linear {r2(lin, lam):.2f}  linear+relu {r2(np.c_[lin, np.maximum(lin,0)], lam):.2f}  "
              f"record-before {r2(rec, lam):.2f}  both {r2(np.c_[lin, np.maximum(lin,0), rec], lam):.2f}")
