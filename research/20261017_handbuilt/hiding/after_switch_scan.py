"""Counterfactual scan: what does a number right after a record do to the four's references later? Lists with a record
M at r = 2..5 over a previous largest P (P > 0.3, M > 1.1 P, M < 2.5); the number at r + 1 is set to x = s P for a grid
of s; the numbers after r + 1 are set negative (-0.5) so nothing else clips. Reference of neuron n at step t:
ref_n(t) = (W_hh h(t-1))_n / c_n, c_n = -w_hi[n]. Printed: median over lists of (ref_n(r + k) - P) / P, k = 1..4, per s.
If clipping makes the references follow max(P, x), the rows for s > 1 rise with s for the neurons whose reference was P."""
import numpy as np
from common import WI, WH, C, lists, run
rng = np.random.default_rng(6); cn = -WI[C]
Xa = lists(6, 400_000)
for r in (3,):
    p = Xa[:, :r].max(1); ok = (p > 0.3) & (Xa[:, r] > 1.1 * p) & (Xa[:, r] < 2.5) & (Xa[:, :r].argmax(1) == r - 1) | False
    ok = (p > 0.3) & (Xa[:, r] > 1.1 * p) & (Xa[:, r] < 2.5)
    X0 = Xa[ok][:20000].copy(); X0[:, r + 2:] = -0.5; P = X0[:, :r].max(1); Mv = X0[:, r]
    print(f"record at r = {r}, {len(X0)} lists; median M / P = {np.median(Mv / P):.2f}")
    print("   s = x/P   " + "   ".join(f"k={k}: " + " ".join(f"n{n:<5d}" for n in C) for k in range(1, 5)))
    for s in (-0.5, 0.0, 0.5, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4):
        X1 = X0.copy(); X1[:, r + 1] = np.minimum(s * P, 0.999 * Mv)
        _, Pr, H = run(X1); ref = (Pr[:, :, C] - X1[:, :, None] * WI[C]) / cn
        cells = []
        for k in range(1, 5):
            cells.append("      " + " ".join(f"{np.median((ref[:, r + k, i] - P) / P):+6.3f}" for i in range(4)))
        offs = (H[:, r + 1, C] == 0).mean(0)
        print(f"   {s:5.2f}    " + "".join(cells) + "   off at r+1: " + " ".join(f"{v:.2f}" for v in offs))
