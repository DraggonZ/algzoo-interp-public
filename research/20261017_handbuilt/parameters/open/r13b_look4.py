"""r13b look 4: is the zero-sum offset d = (g Push(row_9) - row_0)/9 consistent once the arriving number is unknown?
Push(row_9) through x_new = 0 (as the rule) and through a fresh standard-normal x_new (its expected score).
Also the per-row deficits D_k = row_k - g Push(row_{k-1}) for k = 1..9 projected on row 0."""
import numpy as np
import r12a_x as X

WI, WH, WO = X.WI, X.WH, X.WO
N = 400_000
rg = np.random.default_rng(101)
x = rg.standard_normal((N, 10))
h = np.zeros((N, 16))
for t in range(10):
    h = np.maximum(h @ WH.T + x[:, t, None] * WI[None, :], 0)
G = h.T @ h / N
h11z = np.maximum(h @ WH.T, 0)
xf = rg.standard_normal(N)
h11f = np.maximum(h @ WH.T + xf[:, None] * WI[None, :], 0)
Mz = np.linalg.solve(G, h.T @ h11z / N)
Mf = np.linalg.solve(G, h.T @ h11f / N)
g = 1.0177
r0 = WO[0]; n0 = np.linalg.norm(r0)
for lab, M in [("x_new = 0", Mz), ("x_new fresh", Mf)]:
    P9 = M @ WO[9]
    print(f"{lab}: |P row9|/|row0| {np.linalg.norm(P9) / n0:.2f}, coeff on row0 {P9 @ r0 / n0**2:+.3f}")
    D = np.array([WO[k] - g * M @ WO[k - 1] for k in range(1, 10)])
    print("  deficits' coeff on row0, k=1..9:", (D @ r0 / n0**2).round(3))
    print("  deficits' |.|/|row0|:          ", (np.linalg.norm(D, axis=1) / n0).round(2))
    print("  mean_{1..8} D coeff %+.3f; predicted d = (gP9 - row0)/9 coeff %+.3f" % (D[:8].mean(0) @ r0 / n0**2,
          (g * P9 - r0) @ r0 / n0**2 / 9))
print("ARC column sums: coeff on row0 %+.3f, |sum|/|row0| %.2f" % (WO.sum(0) @ r0 / n0**2, np.linalg.norm(WO.sum(0)) / n0))
