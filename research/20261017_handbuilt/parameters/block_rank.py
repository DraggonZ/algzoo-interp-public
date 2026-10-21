"""(a) Low-rank descriptions of recurrent blocks: replace the block by its rank-r SVD truncation, cost per r.
(b) Relative rounding: every weight rounded on a log scale with step eps, cost per eps; per kind of weight.
Paired accuracy on 2^22 lists (as census.py)."""
import numpy as np
import torch as th
from algzoo.core import example

W0 = [np.asarray(p, float) for p in example()]
dev = "cuda"; N = 2**22
X = th.randn(N, 10, generator=th.Generator(device=dev).manual_seed(7), device=dev)
Y = X.argsort(-1)[:, -2]


@th.no_grad()
def acc(P):
    wi, wh, wo = (th.as_tensor(p, dtype=th.float32, device=dev) for p in P)
    h = th.zeros(N, 16, device=dev)
    for t in range(10):
        h = th.relu(h @ wh.T + X[:, t, None] * wi)
    return ((h @ wo.T).argmax(-1) == Y).float().mean().item()


acc0 = acc(W0)
print(f"ARC {acc0:.5f}")
blocks = {"memory+hiding (12)": [0, 1, 3, 6, 7, 8, 10, 11, 12, 13, 14, 15], "hiding (4)": [1, 6, 7, 8], "all 16": list(range(16))}
for name, B in blocks.items():
    A = W0[1][np.ix_(B, B)]; n = len(B)
    U, s, Vt = np.linalg.svd(A)
    print(f"\n{name}: singular values {np.round(s, 2)}")
    ev = np.linalg.eigvals(A); ev = ev[np.argsort(-abs(ev))]
    print("  eigenvalues (modulus, degrees): " + ", ".join(f"{abs(e):.2f} {np.degrees(np.angle(e)):+.0f}" for e in ev))
    for r in range(1, n + 1):
        P = [w.copy() for w in W0]
        P[1][np.ix_(B, B)] = (U[:, :r] * s[:r]) @ Vt[:r]
        print(f"  rank {r:2d} ({r * (2 * n - r):3d} numbers for {n * n}): cost {100 * (acc0 - acc(P)):6.2f} points")


def rnd(w, eps):
    a = np.abs(w); out = np.zeros_like(w); nz = a > 0
    out[nz] = np.sign(w[nz]) * np.exp(np.round(np.log(a[nz]) / eps) * eps)
    return out


print("\nrelative rounding (log-scale step eps; max relative change about eps/2):")
for eps in [0.002, 0.005, 0.01, 0.02, 0.05, 0.1]:
    row = []
    for k, lab in [(None, "all"), (0, "input"), (1, "recurrent"), (2, "readout")]:
        P = [rnd(w, eps) if (k is None or k == m) else w.copy() for m, w in enumerate(W0)]
        row.append(f"{lab} {100 * (acc0 - acc(P)):6.2f}")
    print(f"  eps {eps:.3f}: cost in points: " + ", ".join(row))
