"""The four hiding neurons' own block W_cc (weights among 1, 6, 7, 8): eigenvalues (size, angle per step), and what it
does to the arriving number's push (their input weights) and to the store's push (their weights from neuron 2): if
these land in the fast-dying directions, a number's direct term cancels across the group one step later and the level
is supplied afresh every step. Also c_n = minus the input weights."""
import numpy as np
from common import WI, WH, C
W = WH[np.ix_(C, C)]; ev = np.linalg.eigvals(W)
print("c (minus input weights) for neurons 1, 6, 7, 8:", " ".join(f"{-WI[n]:.2f}" for n in C))
print("eigenvalues of the block: " + ", ".join(f"size {abs(e):.3f} angle {np.degrees(np.angle(e)):+.1f} deg" for e in ev))
for lab, v in (("input weights", WI[C]), ("weights from the store (2)", WH[C, 2]), ("weights from the detector (4)", WH[C, 4])):
    print(f"{lab:30s}: {np.round(v, 3)}, size {np.linalg.norm(v):.2f}; after one step of the block {np.round(W @ v, 3)}, size {np.linalg.norm(W @ v):.2f}")
