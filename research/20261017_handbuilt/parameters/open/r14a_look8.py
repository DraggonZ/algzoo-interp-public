"""r14a look 8: where does the smallest-arrival push gain? Split its change to rows 1-8 (relative to the zero-arrival
push) into the part common to all eight rows and the row-specific rest; also the detector column alone.
Priced with r14a's structure bits."""
import numpy as np
import torch as th
import r12a_x as X

R = X.load("rules/r14a_minarrival.py")
B = R.B
D = R._REG - B.B._REG          # [8, 16], before the gain
common = D.mean(0)
spec = D - common
print("mean change over rows 1-8 per column:", np.round(common, 2))
print("row-specific rms %.3f, common rms %.3f" % (np.sqrt((spec ** 2).mean()), np.sqrt((common ** 2).mean())))


class V:
    def __init__(self, DD):
        self.COVERED = R.COVERED; self.ORDER = R.ORDER; self.KINDS = R.KINDS
        self.STRUCTURE_BITS = R.STRUCTURE_BITS; self.init_constants = R.init_constants
        self.d = th.zeros(len(R.COVERED), dtype=th.float64)
        for j, n in enumerate(R.COVERED):
            s, t = n.split("->")
            if t in {f"out{k}" for k in range(1, 9)}:
                self.d[j] = DD[int(t[3:]) - 1, int(s)]

    def weights(self, C):
        return B.weights(C) + C[0] * self.d


det = np.zeros_like(D); det[:, 4] = D[:, 4]
for lab, DD in [("full change", D), ("common part only", np.tile(common, (8, 1))), ("row-specific only", spec),
                ("detector column only", det)]:
    print(f"{lab:22s} {X.price(V(DD))[0]:+.1f}")
