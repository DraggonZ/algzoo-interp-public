"""Is the keep gate's history part D5 the memory's linear echo of earlier writes?
(1) The fitted table (g16_tables.pkl, D5 over m1/m2 by ages + past hinge values + past gate values) next to the echo
    predicted from the weights alone if all eight memory neurons stayed on: a write w_j(s) by neuron j at step s reaches
    neuron 5's pre-activation at step t+1 through W_5,mem A^(t-s-1) W_mem,j (A = memory-to-memory weights), plus the
    direct weight when s = t (only the gates' own values at age 1). Units: per unit of hinge value (neuron value / |w_j|).
(2) Weight-only D5: run the memory linearly (no ReLU) on the network's own other-neuron values and compare with the
    true D5; same with the memory's real switching (identity check)."""
import pickle
import numpy as np
from gatelib import WI, WH, MEM, lists, run
from gaterules3 import HJ

co = pickle.load(open("g16_tables.pkl", "rb"))
c = co[("D5", "gates|-kept|-forgot|-neg|-rec|-p")]
n_roles = 9 + 72
hz = c[n_roles + 3:n_roles + 3 + 45].reshape(9, 5); gz = c[n_roles + 3 + 45:].reshape(9, 2)
A = WH[np.ix_(MEM, MEM)]; w5m = WH[5, MEM]
print("fitted coefficient of each past hinge value on D5, by age a (rows) and neuron (columns), next to the all-on echo")
print("      " + "".join(f"{f'fit {j}':>8s}{f'echo {j}':>8s}" for j in HJ))
for a in range(1, 10):
    row = ""
    for k, j in enumerate(HJ):
        echo = w5m @ np.linalg.matrix_power(A, a - 1) @ WH[MEM, j] * abs(WI[j]) if a >= 1 else 0
        row += f"{hz[a - 1, k]:8.2f}{echo:8.2f}"
    print(f"a={a}  " + row)
print("fitted coefficient of past gate values (one step after the number of age a), next to the echo")
for a in range(1, 10):
    out = []
    for k, g in enumerate((5, 9)):
        echo = w5m @ np.linalg.matrix_power(A, a - 2) @ WH[MEM, g] if a >= 2 else 0.0
        out.append(f"gate {g}: fit {gz[a - 1, k]:6.2f} echo {echo:6.2f}")
    print(f"a={a}  " + "   ".join(out))
print(f"own previous value: fit {c[n_roles + 2]:.3f} (direct weight {WH[5, 5]:.2f}); x_t {c[n_roles]:.3f}, x_(t+1) {c[n_roles + 1]:.3f} (direct {WI[5]:.2f})")

X = lists(21, 100_000); L, P, H = run(X)
oth = [j for j in range(16) if j not in MEM]
hm = np.zeros((len(X), 8)); err = []; err_true = []
for t in range(10):
    hm = hm @ A.T + (H[:, t - 1][:, oth] @ WH[np.ix_(MEM, oth)].T if t > 0 else 0) + X[:, t:t + 1] * WI[MEM]
    if t <= 8 and t >= 1:
        Dtrue = P[:, t + 1, 5] - sum(WH[5, j] * H[:, t, j] for j in HJ)
        Dlin = hm @ w5m + sum(WH[5, j] * H[:, t, j] for j in oth if j not in HJ) + WI[5] * X[:, t + 1]
        Dre = H[:, t][:, MEM] @ w5m + sum(WH[5, j] * H[:, t, j] for j in oth if j not in HJ) + WI[5] * X[:, t + 1]
        err.append(Dlin - Dtrue); err_true.append(Dre - Dtrue)
err = np.concatenate(err); err_true = np.concatenate(err_true)
print(f"\nD5 from a linear memory run on the network's own writes: rms error {np.sqrt((err ** 2).mean()):.2f} (D5 sd about 6.5); with the real memory: {np.sqrt((err_true ** 2).mean()):.1e}")
