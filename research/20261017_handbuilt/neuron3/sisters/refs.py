"""Hiding references, refitted. Reference of hiding neuron j at step t = -(W_hh[j] . h_{t-1}) / w_j (the arriving number at
which j would switch). Fitted on the largest and second-largest seen before step t (steps 2-9), as in quantities.py.
quantities.py's fit is broken for 12 hiding neurons in 5 networks by rare huge negative references (down to -200);
here the fit is also repeated on the middle 98% of references (trimmed). Writes refs.log."""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from sisters import *

out = []
for lab, W in networks().items():
    hid = ROLES[lab]["hiding"]
    if not hid: continue
    X = lists(3, 100_000); _, P, H = run(W, X); V = task(X); line = []
    for j in hid:
        y = (-(H[:, 1:9] @ W[1][j]) / W[0][j]).ravel()                       # comparison at steps 2..9
        A = np.c_[V["m1"][:, 1:9].ravel(), V["m2"][:, 1:9].ravel(), np.ones(y.size)]
        lo, hi = np.percentile(y, [1, 99]); k = (y > lo) & (y < hi)
        c = np.linalg.lstsq(A[k], y[k], rcond=None)[0]; r2 = 1 - ((A[k] @ c - y[k]) ** 2).mean() / y[k].var()
        line.append(f"{j}: {c[0]:.2f} x largest + {c[1]:.2f} x second + {c[2]:.2f} (R2 {r2:.2f})")
    s = f"{lab:6} " + "; ".join(line); print(s, flush=True); out.append(s)
open(os.path.join(HERE, "refs.log"), "w").write("Trimmed fits (middle 98%) of hiding references on largest and second-largest seen so far, steps 2-9\n" + "\n".join(out) + "\n")
