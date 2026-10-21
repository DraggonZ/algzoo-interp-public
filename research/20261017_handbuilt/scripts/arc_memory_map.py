"""Map the trained network's memory: each neuron's readout over positions, its weights to/from the other memory
neurons, and its eigen-structure. Memory = 0, 3, 10-15."""
import numpy as np
from algzoo.core import example
wi, wh, wo = (np.asarray(w, dtype=float) for w in example()); wi = wi.reshape(-1)
mem = [0, 3, 10, 11, 12, 13, 14, 15]
np.set_printoptions(precision=2, suppress=True, linewidth=170)
print("readout of each memory neuron over positions 0-9 (rows)")
for k in mem: print(f"  {k:2d}: {wo[:, k]}")
print("memory -> memory weights (row = to)"); print(wh[np.ix_(mem, mem)])
print("inputs to memory from cancellers 1,6,7,8 | store 2, detector 4, keep 5, late 9")
print(np.c_[wh[np.ix_(mem, [1, 6, 7, 8])], wh[np.ix_(mem, [2, 4, 5, 9])]])
M = wh[np.ix_(mem, mem)]; ev, V = np.linalg.eig(M)
print("eigenvalues of memory->memory (all on): modulus, angle deg"); 
for e in sorted(ev, key=lambda z: -abs(np.angle(z))): print(f"  {abs(e):.2f} {np.degrees(np.angle(e)):7.1f}")
lin = [10, 11, 12, 13, 14]; Ml = wh[np.ix_(lin, lin)]; ev = np.linalg.eigvals(Ml)
print("eigenvalues of the always-on part 10-14 alone:", " ".join(f"{abs(e):.2f}@{np.degrees(np.angle(e)):.0f}" for e in ev))
