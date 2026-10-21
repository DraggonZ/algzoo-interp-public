"""How rough is accuracy as a function of the weights? A limit for estimators that expand around ARC's model.

For each noise level, make N noisy copies (each weight times 1 + sigma*z), measure accuracy by sampling (2^21 shared
inputs, so copy-to-copy differences have little sampling noise), then fit on 2/3 of the copies and test on 1/3:
  linear in z (432 numbers), and linear plus quadratic in the 36 weights among the comparing neurons.
These fits use true accuracies, so they are not estimators: they are the best any estimator of that form could do.
Output: reports/roughness_{sigma}.npz and a log line per fit.
"""
import sys
import time

import numpy as np
import torch as th

from algzoo.core import example

dev = "cuda"
N = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
S_IN = 2 ** 21
w_hi, w_hh, w_oh = example()
base = [th.as_tensor(w, device=dev, dtype=th.float32) for w in (w_hi, w_hh, w_oh)]
g = th.Generator(device=dev).manual_seed(123)
X = th.randn(S_IN, 10, device=dev, generator=g)
tgt = X.argsort(-1)[:, -2]


@th.no_grad()
def accs(Z):
    """Z: [C, 432] perturbations -> accuracies [C]."""
    C = Z.shape[0]
    hi = base[0] * (1 + Z[:, :16])
    hh = base[1] * (1 + Z[:, 16:272].reshape(C, 16, 16))
    oh = base[2] * (1 + Z[:, 272:].reshape(C, 10, 16))
    right = th.zeros(C, device=dev)
    for i in range(0, S_IN, 2 ** 15):
        x = X[i:i + 2 ** 15]
        h = th.zeros(C, x.shape[0], 16, device=dev)
        for t in range(10):
            h = th.relu(th.bmm(h, hh.transpose(1, 2)) + x[None, :, t, None] * hi[:, None, :])
        pred = th.bmm(h, oh.transpose(1, 2)).argmax(-1)
        right += (pred == tgt[i:i + 2 ** 15][None]).float().sum(1)
    return (right / S_IN).cpu().numpy()


C_IDX = [1, 2, 4, 6, 7, 8]
comp = [16 + i * 16 + j for i in C_IDX for j in C_IDX]


def ridge(A, y, lam):
    A1 = np.c_[np.ones(len(A)), A]
    R = lam * np.eye(A1.shape[1]); R[0, 0] = 0
    return np.linalg.solve(A1.T @ A1 + R, A1.T @ y)


for sigma in (0.005, 0.01, 0.02):
    t0 = time.time()
    rng = np.random.default_rng(int(sigma * 1e4))
    Z = rng.standard_normal((N, 432)).astype(np.float32)
    A = np.concatenate([accs(th.as_tensor(sigma * Z[i:i + 64], device=dev)) for i in range(0, N, 64)])
    np.savez(f"/data/output/research/algzoo/20261016_run3/reports/roughness_{sigma}.npz", Z=Z, acc=A, sigma=sigma)
    ntr = 2 * N // 3
    Q = np.stack([Z[:, a] * Z[:, b] for k, a in enumerate(comp) for b in comp[k:]], 1)
    feats = {"linear in all 432": Z, "linear + quadratic in 36 comparing weights": np.c_[Z, Q]}
    sd = A.std()
    line = [f"sigma {sigma}: {N} copies, accuracy mean {A.mean():.3f}, spread (SD) {sd:.4f}, {time.time()-t0:.0f}s"]
    for lab, F in feats.items():
        best = None
        for lam in (0.1, 1, 10, 100, 1000):
            w = ridge(F[:ntr], A[:ntr], lam)
            pred = np.c_[np.ones(N - ntr), F[ntr:]] @ w
            rmse = np.sqrt(np.mean((pred - A[ntr:]) ** 2))
            if best is None or rmse < best[0]:
                best = (rmse, lam)
        line.append(f"   {lab}: held-out RMSE {best[0]:.4f} (R^2 {1 - best[0]**2 / A[ntr:].var():.2f}, ridge {best[1]})")
    print("\n".join(line), flush=True)
