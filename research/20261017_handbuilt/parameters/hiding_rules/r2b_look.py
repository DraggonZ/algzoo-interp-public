"""r2b diagnostic: the masked rule's one-step constants next to ARC's weights into the hiding neurons."""
import sys
sys.path.insert(0, "rules")
import numpy as np
from common import HID, WI, WH
import simlib
import r1b_masked as R

X = simlib.lists(101, 50_000); _, P, H = simlib.run(X)
C = R.init_constants(X, P, H)
M = np.zeros(40); M[np.flatnonzero(R.FULL.ravel())] = C; M = M.reshape(4, 10)
np.set_printoptions(precision=3, suppress=True, linewidth=150)
print("features", R.FEATS); print(M[:, :6])
print("feedback (row: reader 1, 6, 7, 8; column: source 1, 6, 7, 8)"); print(M[:, 6:])
print("ARC weights source -> reader, same layout"); print(WH[np.ix_(HID, HID)])
print("input weights", WI[HID])
print("ARC weights from 2, 4, 5, 9 into the hiding neurons"); print(WH[np.ix_(HID, [2, 4, 5, 9])])
print("store/detector block", WH[np.ix_([2, 4], [2, 4])], WI[[2, 4]])
print("mean activity", H.mean((0, 1)).round(2))
