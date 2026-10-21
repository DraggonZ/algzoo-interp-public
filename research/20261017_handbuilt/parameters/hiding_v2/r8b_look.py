"""r8b: look at the hiding rows (ARC's weights) and what the current best leaves uncovered. Diagnostic only."""
import numpy as np
from algzoo.core import example
np.set_printoptions(precision=3, suppress=True, linewidth=160)
WI, WH, WO = example()
H = [1, 6, 7, 8]
print("input weights", {j: round(float(WI[j]), 3) for j in range(16)})
print("rows of hiding neurons (cols 0..15):")
for j in H: print(j, WH[j])
A = np.array([[WH[j, i] * WI[i] / WI[j] for i in H] for j in H])
print("hiding block in reference units A[j,i]=W[j,i] w_i/w_j (rows 1,6,7,8):\n", A)
print("raw hiding block W[j,i]:\n", WH[np.ix_(H, H)])
print("eig raw", np.linalg.eigvals(WH[np.ix_(H, H)]))
print("columns hiding -> others (rows j, cols 1,6,7,8):")
for j in range(16): print(j, WH[j, H])
print("readout from hiding:\n", WO[:, H])
