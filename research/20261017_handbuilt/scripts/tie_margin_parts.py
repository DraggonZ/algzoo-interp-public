"""ARC, reduced inputs: largest m1 at p1, second and third tied (m2 = m3 = v) at p2, p3,
seven others at background b. Scores are exactly J x (no biases), so the tie margin
M = s[p2] - s[p3] splits into parts. Which part varies across the 720 placements?"""
import itertools

import numpy as np
import torch as th

from algzoo.core import example, forward

P = example()
m1, v, bg = 1.6, 1.0, -0.46
rows = []
xs, pl = [], []
for p1, p2, p3 in itertools.permutations(range(10), 3):
    x = np.full(10, bg); x[p1] = m1; x[p2] = v; x[p3] = v * (1 - 1e-6)
    xs.append(x); pl.append((p1, p2, p3))
X = th.tensor(np.array(xs))
J = th.func.vmap(th.func.jacrev(lambda z: forward(P, z[None])[0]))(X).numpy()  # [N, slot, input]
pl = np.array(pl)
N = len(pl)
r = np.arange(N)
p1, p2, p3 = pl.T
d = J[r, p2] - J[r, p3]  # margin row: d[k] = dM/dx_k
leak = d[r, p1] * m1
own = J[r, p2, p2] * v - J[r, p3, p3] * v
cross = J[r, p2, p3] * v - J[r, p3, p2] * v
bgpart = (d.sum(1) - d[r, p1] - d[r, p2] - d[r, p3]) * bg
M = leak + own + cross + bgpart
s = forward(P, X).numpy()
print("max |parts - true margin|:", np.abs(M - (s[r, p2] - s[r, p3])).max())
print(f"margin at the tie: mean {M.mean():.1f}, sd {M.std():.1f}; placements where the third wins: {np.mean(M < 0):.2f}")
parts = {"largest's leak": leak, "own votes": own, "crosstalk": cross, "background": bgpart}
for k, p in parts.items():
    print(f"  {k:15s} mean {p.mean():7.1f}  sd {p.std():6.1f}  corr with margin {np.corrcoef(p, M)[0,1]:5.2f}")
# leave-one-part-out: how much of the margin's variance is lost when a part is replaced by its mean
for k, p in parts.items():
    Mk = M - p + p.mean()
    print(f"  without variation in {k:15s}: sign of margin agrees in {np.mean(np.sign(Mk) == np.sign(M)):.2f} of placements")
np.savez("tie_parts.npz", pl=pl, M=M, leak=leak, own=own, cross=cross, bg=bgpart, J=J)
