"""Do the references track the running second-largest? On all lists (any numbers since the record), per (neuron, K)
pooled over record positions r = 1..(9-K), fit c_n ref_n(t) by least squares on (a) previous largest P and largest M,
(b) the second-largest so far m2 (= max of P and the numbers since the record) and M, (c) P, m2 and M. Report the
residual RMS in pre-activation units and, for (c), the reference's weights on P, m2 and M (divided by c_n).
Restricted to lists with P > 0.2 so that a second-largest exists. 300,000 lists, seed 9."""
import numpy as np
from common import WI, WH, C, lists, run
from rulelib import list_vars, CN
X = lists(9, 300_000); V = list_vars(X); _, P, H = run(X); T = P[:, :, C] - X[:, :, None] * WI[C]
for K in range(1, 6):
    rows = []
    for rr in range(1, 10 - K):
        t = rr + K; idx = np.where((V["R"][:, t] == rr) & (V["P"][:, t] > 0.2))[0]
        rows.append(np.c_[V["P"][idx, t], V["M2"][idx, t], V["M"][idx, t], T[idx, t]])
    R = np.concatenate(rows); out = []
    frac = np.mean(R[:, 1] > R[:, 0] + 1e-9)
    for i, n in enumerate(C):
        y = R[:, 3 + i]; res = {}
        for lab, cols in (("P,M", [0, 2]), ("m2,M", [1, 2]), ("P,m2,M", [0, 1, 2])):
            A = R[:, cols]; co = np.linalg.lstsq(A, y, rcond=None)[0]; res[lab] = (np.sqrt(np.mean((y - A @ co) ** 2)), co / CN[i])
        w = res["P,m2,M"][1]
        out.append(f"n{n}: rms P,M {res['P,M'][0]:.2f} | m2,M {res['m2,M'][0]:.2f} | P,m2,M {res['P,m2,M'][0]:.2f} (weights P {w[0]:+.2f} m2 {w[1]:+.2f} M {w[2]:+.2f})")
    print(f"K={K} (n={len(R)}, second-largest above P on {frac:.2f}):\n   " + "\n   ".join(out))
