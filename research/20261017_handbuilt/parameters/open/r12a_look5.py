"""r12a look 5: the regression push's residual along readout row 0. Per row: a(k) = least-squares coefficient of the
residual on row 0 (push columns); cosine with (row 0 - push(row 0)); and a gauge form (A): rows measured relative
to row 0, r_k - r_0 = push(r_{k-1} - r_0)."""
import numpy as np
import r12a_look3 as L
X = L.X
WO = X.WO
T = L.TGT
ALL = list(range(16))
cos = lambda a, b: a @ b / np.linalg.norm(a) / np.linalg.norm(b)
N = L.N
h11z = np.maximum(L.h10 @ X.WH.T, 0)
reg = lambda y: np.linalg.solve(L.G, L.h10.T @ y / N)
p0 = reg(h11z @ WO[0])
for k in range(1, 10):
    r = L.REG["zero", k]
    e = WO[k] - r
    a = (e[T] @ WO[0, T]) / (WO[0, T] @ WO[0, T])
    rA = r - p0 + WO[0]
    eA = np.linalg.norm((WO[k] - rA)[T]) / np.linalg.norm(WO[k, T])
    print(f"row {k}: a(k) {a:+.3f}; residual left after a(k) row0 {np.linalg.norm((e - a * WO[0])[T]) / np.linalg.norm(WO[k, T]):.3f} "
          f"(was {np.linalg.norm(e[T]) / np.linalg.norm(WO[k, T]):.3f}); cos(e, r0 - push r0) {cos(e[T], (WO[0] - p0)[T]):+.2f}; gauge-A err {eA:.3f}")
print("row 0 vs its own push:", np.round(WO[0, T], 2), np.round(p0[T], 2))
