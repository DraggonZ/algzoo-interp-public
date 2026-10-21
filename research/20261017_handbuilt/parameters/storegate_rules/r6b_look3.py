"""r6b look 3: is the detector's recurrent drive a linear read of the running largest from the state?
Run ARC on random lists; at each step t >= 1 take the state h_t (before x_t is added), m1 and m2 (largest and
second-largest of x_0..x_{t-1}). Fit u4 = W[4] . h_t by a m1 + b m2 (how well does it hold?), then solve the row
from the states: row = argmin |H row + k (a m1 + b m2)|^2 and compare with ARC's row 4 (rms per weight).
Same for the store row against m1 alone (the store is one step behind: compare with the largest before x_{t-1})."""
import numpy as np
import torch as th
from algzoo.core import example

WI, WH, WO = example()
rng = np.random.default_rng(101)
N = 200000
X = rng.standard_normal((N, 10))
h = np.zeros((N, 16)); H, M1, M2, M1lag, T = [], [], [], [], []
for t in range(10):
    if t >= 1:
        prev = X[:, :t]
        s = np.sort(prev, 1)
        H.append(h.copy()); M1.append(s[:, -1]); M2.append(s[:, -2] if t >= 2 else np.zeros(N)); T.append(np.full(N, t))
        M1lag.append(np.sort(X[:, :t - 1], 1)[:, -1] if t >= 2 else np.zeros(N))
    h = np.maximum(h @ WH.T + X[:, t, None] * WI, 0)
H, M1, M2, M1lag, T = map(np.concatenate, (H, M1, M2, M1lag, T))
k = WI[4]
u4 = H @ WH[4]
for name, F in [("m1", [M1]), ("m1, m2", [M1, M2])]:
    A = np.stack(F, 1); co, *_ = np.linalg.lstsq(A, u4, rcond=None)
    r = u4 - A @ co
    print(f"u4 ~ {name}: coefs/k {np.round(co / -k, 4)}, R2 {1 - r.var() / u4.var():.5f}, resid rms {r.std():.3f} (u4 rms {u4.std():.2f})")
    row, *_ = np.linalg.lstsq(H, A @ co, rcond=None)
    print("   row solved from states:", np.round(row, 3))
    print("   ARC row 4:            ", np.round(WH[4], 3))
    print(f"   rms diff {np.sqrt(((row - WH[4]) ** 2).mean()):.4f}")
u2 = H @ WH[2]
for name, F in [("m1", [M1]), ("m1, m1lag", [M1, M1lag])]:
    A = np.stack(F, 1); co, *_ = np.linalg.lstsq(A, u2, rcond=None)
    r = u2 - A @ co
    print(f"u2 ~ {name}: coefs {np.round(co, 3)}, R2 {1 - r.var() / u2.var():.5f}")
# conditioning of H: singular values
print("state singular values", np.round(np.linalg.svd(H / np.sqrt(len(H)), compute_uv=False), 3))
