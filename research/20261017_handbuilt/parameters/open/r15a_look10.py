"""r15a look 10: where the contender restriction gains. Its change to rows 1-8 (relative to r14a's push) split into
the part common to the eight rows and the row-specific rest; priced under v2 and v2c with r15a's structure bits.
Also the per-row relative error (centred) and the KL-relevant check: the change's size along each row."""
import numpy as np
import torch as th
import r12a_x as X2
import r15a_xc as XC

R = X2.load("rules/r15a_contender.py")
B = R.B
D = R._REG - B._REG
common = D.mean(0); spec = D - common
np.set_printoptions(precision=2, suppress=True, linewidth=200)
print("change rms per row:", np.sqrt((D ** 2).mean(1)))
print("common part:", common)


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


for lab, DD in [("full change", D), ("common part only", np.tile(common, (8, 1))), ("row-specific only", spec)]:
    v = V(DD)
    print(f"{lab:22s} v2 {X2.price(v)[0]:+.1f}  v2c {XC.price(v)[0]:+.1f}")
