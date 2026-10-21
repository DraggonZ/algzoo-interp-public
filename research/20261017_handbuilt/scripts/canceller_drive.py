"""Do ARC's cancellers drive the timer every step? Lone 1.0 at p = 2 (fillers -0.5). Exact contribution of the
cancellers' activity at each step tau (propagated through the list's own on/off pattern) to the final scores, split
into what it does through the timer neurons (all paths that leave the cancellers into other neurons) versus the
cancellers' own direct readout at the end (tau = 9 only)."""
import numpy as np
from algzoo.core import example

wi, wh, wo = (np.asarray(w, dtype=float) for w in example()); wi = wi.reshape(-1)
canc = [1, 6, 7, 8]
np.set_printoptions(precision=0, suppress=True, linewidth=150)
x = np.full(10, -0.5); x[2] = 1.0
hs, masks = [], []
h = np.zeros(16)
for v in x:
    pre = wh @ h + wi * v; masks.append(pre > 0); h = np.maximum(pre, 0); hs.append(h)
for tau in range(2, 10):
    v = np.zeros(16); v[canc] = hs[tau][canc]
    if tau < 9:
        u = masks[tau + 1] * (wh @ v)          # one step into the network
        u[canc] = 0                             # keep only what enters non-canceller neurons
        for t in range(tau + 2, 10): u = masks[t] * (wh @ u)
        s = wo @ u
    else:
        s = wo @ v
    s = s - s.mean()
    print(f"cancellers at step {tau}: activity {np.round(hs[tau][canc], 1)}  -> via other neurons: {s}" if tau < 9 else f"cancellers at step 9, direct readout: {s}")
