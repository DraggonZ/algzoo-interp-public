"""What happens in ARC on the steps after a new largest? Lists: fillers -0.5, old largest 1.0 at position 2,
new largest x at position 5 (x from 1.02 to 2.0). Activity of the detector (4), the gate (5), neuron 8 and the
cancellers' mean, at steps 5, 6, 7. Also a non-record 0.9 at 5 (a new second) for comparison."""
import numpy as np
from algzoo.core import example

wi, wh, wo = (np.asarray(w, dtype=float) for w in example()); wi = wi.reshape(-1)


def states(x):
    h = np.zeros(16); H = []
    for v in x: h = np.maximum(wh @ h + wi * v, 0); H.append(h)
    return np.array(H)


print(f"{'x at 5':>8s} | {'n4 @5':>6s} {'n5 @6':>6s} {'n5 @7':>6s} | {'n8 @5':>6s} {'n8 @6':>6s} | cancellers mean @5 @6 @7")
for xv in (0.9, 1.02, 1.1, 1.3, 1.6, 2.0):
    x = np.full(10, -0.5); x[2] = 1.0; x[5] = xv
    H = states(x)
    c = H[:, [1, 6, 7, 8]].mean(1)
    print(f"{xv:8.2f} | {H[5, 4]:6.2f} {H[6, 5]:6.2f} {H[7, 5]:6.2f} | {H[5, 8]:6.2f} {H[6, 8]:6.2f} | {c[5]:6.2f} {c[6]:6.2f} {c[7]:6.2f}")
