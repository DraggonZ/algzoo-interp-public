"""r8b look 3: is each hiding row's hiding-block part (raw W[j, H]) orthogonal to some other vector on the hiding
group, besides the input weights? Candidates: the column k of W restricted to the hiding rows, W[H, k], for all k.
Also: does a row equal a mix of fixed things. Diagnostic only."""
import numpy as np
from algzoo.core import example
np.set_printoptions(precision=3, suppress=True, linewidth=160)
WI, WH, WO = example()
H = [1, 6, 7, 8]
B = WH[np.ix_(H, H)]
cos = lambda a, b: a @ b / np.linalg.norm(a) / np.linalg.norm(b)
print("cos of each hiding row's block part with W[H,k] (k = source column):")
for k in range(16):
    v = WH[H, k]
    print(f"k={k:2d} v={v}  cos rows 1,6,7,8: " + " ".join(f"{cos(B[r], v):+.3f}" for r in range(4)))
v = WI[H]; print("input:", " ".join(f"{cos(B[r], v):+.3f}" for r in range(4)))
# the block's null spaces / structure
U, s, Vt = np.linalg.svd(B); print("singular values", s); print("right sing vecs\n", Vt)
print("w_in on H normalised", WI[H] / np.linalg.norm(WI[H]))
# A in reference units, left/right eigen
A = B * WI[H][None, :] / WI[H][:, None]
ev, V = np.linalg.eig(A); print("A eig", ev); print(V)
# row 8 and row 1 relation
print("row1/row8 raw", B[0] / B[3])
print("B @ w_in (should be ~0 by blindness):", B @ WI[H])
print("w_in @ B (column sums weighted):", WI[H] @ B)
print("B squared", B @ B)
