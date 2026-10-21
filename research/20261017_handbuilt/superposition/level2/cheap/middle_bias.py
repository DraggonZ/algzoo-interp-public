"""TLQ's systematic error: wrong stretches in the middle of a line (touching neither tie end) are not found.
Same design points (m = 4, seed 0), full walk along each line versus the two end stretches; difference in accuracy
points, and the full walk's cost."""
import sys, numpy as np, torch
sys.path.insert(0, "../../../family")
from common import ARC, trained
import tlq
from linelib import *
Phi = torch.special.ndtr
tr = np.load("../../../hilton/propagate/across.npy")[:, 0]
names = ["ARC"] + [str(s) for s in list(range(100, 116)) + list(range(200, 216))]
out = []
for i in [0, 1, 4, 9, 15, 17, 20, 25, 30]:
    W = ARC if i == 0 else trained(int(names[i]))
    a, fl, P = tlq.estimate(W, 4, 0, return_parts=True)
    X, A, L, T = P["X"], P["A"], P["L"], P["T"]; n = len(X); r = torch.arange(n, device=DEV); pS = A[:, 1]
    X0 = X.clone(); X0[r, pS] = T
    (rr, lo, hi), _, flw = walk(W, X0, pS, pS, L - T)
    right = torch.zeros(n, device=DEV, dtype=dt).index_add_(0, rr, Phi(T[rr] + hi) - Phi(T[rr] + lo)) / (Phi(L) - Phi(T))
    a_full = right.mean().item()
    out.append((a_full - a) * 100)
    print(f"{names[i]}: ends only {100*a:.3f}, full walk {100*a_full:.3f}, middle stretches cost {100*(a-a_full):+.3f} points (true {100*tr[i]:.3f}); full-walk cost {flw/F_LIST/len(X):.1f} lists per line vs ends {fl/F_LIST/len(X):.1f}", flush=True)
print(f"middle-stretch bias: mean {-np.mean(out):+.3f} points, range {-np.max(out):+.3f} to {-np.min(out):+.3f}")
