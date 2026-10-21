"""r8b look 1: per-row errors of the current best, value of each hiding row, and of perfect knowledge of what is left."""
import numpy as np
import r8b_chainx as X
S, WI, WH = X.S, X.WI, X.WH
cur, C = X.rule_weights(X.CURRENT)
print("constant", C)
g0 = X.gain(cur); print(f"current gain {g0:+.1f}")
X.row_report(cur)
H = [1, 6, 7, 8]
for j in H:  # drop row j's hiding statements
    p = {n: v for n, v in cur.items() if S.row_of(n) != f"in{j}"}
    print(f"without row {j}: {X.gain(p) - g0:+.1f}")
# perfect values on uncovered hiding-row weights (upper bound of what statements could add)
unc = [f"{i}->{j}" for j in H for i in range(16) if f"{i}->{j}" not in cur]
print("uncovered:", unc)
p = dict(cur); p.update({n: S.w_flat[S.IDX[n]] + 1e-3 for n in unc}); print(f"all uncovered exact(1e-3): {X.gain(p) - g0:+.1f}")
for e in (0.01, 0.03, 0.05):
    rng = np.random.default_rng(0); p = dict(cur); p.update({n: S.w_flat[S.IDX[n]] + e * rng.standard_normal() for n in unc})
    print(f"uncovered at rms {e}: {X.gain(p) - g0:+.1f}")
# row 7 exact in hiding cols only
p = dict(cur); p.update({f"{i}->7": WH[7, i] for i in H}); print(f"row 7 hiding cols exact: {X.gain(p) - g0:+.1f}")
p = dict(cur); p.update({f"{i}->7": WH[7, i] for i in range(16)}); print(f"row 7 all exact: {X.gain(p) - g0:+.1f}")
print("row7/row4 ratios:", np.round(WH[7] / WH[4], 2))
print("row 4:", np.round(WH[4], 3))
print("w7/w4", WI[7] / WI[4])
