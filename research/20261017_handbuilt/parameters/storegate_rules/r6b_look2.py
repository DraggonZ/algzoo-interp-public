"""r6b look 2: the detector row's residual after -c x store row. Does one more stated vector explain it?
Least squares row4 ~ a row2 + b X over the 16 recurrent columns, X = every other row, column, readout row,
and the unit vectors (a single weight off). Reports rms residual (base: X absent)."""
import numpy as np
from algzoo.core import example

WI, WH, WO = example()
r2, r4 = WH[2], WH[4]
def fit(X):
    A = np.stack([r2] + X, 1)
    co, *_ = np.linalg.lstsq(A, r4, rcond=None)
    return np.sqrt(((A @ co - r4) ** 2).mean()), co
print("base", fit([]))
cands = [(f"row {k}", WH[k]) for k in range(16) if k not in (2, 4)] + [(f"col {k}", WH[:, k]) for k in range(16)]
cands += [(f"out{k}", WO[k]) for k in range(10)] + [("input", WI)] + [(f"e{k}", np.eye(16)[k]) for k in range(16)]
res = sorted([(fit([x])[0], n, fit([x])[1]) for n, x in cands])
for r in res[:12]:
    print(f"{r[1]:8s} rms {r[0]:.4f} coefs {np.round(r[2], 3)}")
# the same for the store row itself: is row 2 a multiple of another row plus little?
print("row2 vs row7:", np.round(WH[2] / WH[7], 2))
print("row2", np.round(WH[2], 3)); print("row7", np.round(WH[7], 3))
