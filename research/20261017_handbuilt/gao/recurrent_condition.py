"""Recurrent version of the per-neuron condition number. Backward pass through the actual run (ReLU on/off pattern of
each list fixed): g_9 = w_o[winner] - w_o[runner-up], g_{t-1} = W_hh^T (mask_t * g_t). Term T_{t,n} = g_{t,n} h_{t,n}:
first-order change of the lead when neuron n's value at step t is scaled by (1 + e) is e T_{t,n}.
Per neuron: median over lists of sum_t |T_{t,n}| / lead (recurrent condition), and the first-order prediction of the
2%-scale loss (share of right lists where 0.02 sum_t T_{t,n} < -lead, minus wrong lists it would fix).
Compared with the measured losses in neuron_condition.log (same lists, seed 2222)."""
import re
import numpy as np
from scipy.stats import spearmanr
import simlib

X = simlib.lists(2222, 100_000); y = X.argsort(1)[:, -2]; r = np.arange(len(X))
L, P, H = simlib.run(X); k = L.argmax(1); j = np.argsort(L, 1)[:, -2]; lead = L[r, k] - L[r, j]
g = simlib.WO[k] - simlib.WO[j]; T = np.zeros_like(H)
for t in range(9, -1, -1):
    T[:, t] = g * H[:, t]
    g = (g * (P[:, t] > 0)) @ simlib.WH
readout = np.median(np.abs(T[:, 9]) / lead[:, None], 0); recur = np.median(np.abs(T).sum(1) / lead[:, None], 0)
S = T.sum(1)  # first-order lead change per unit relative scale of neuron n at every step
right = k == y
pred = np.array([np.mean(right & (lead + 0.02 * S[:, n] < 0)) for n in range(16)])
log = open("neuron_condition.log").read()
meas = np.zeros(16)
for m in re.finditer(r"neuron\s+(\d+): push.*?2% too large (-?[0-9.]+)", log):
    meas[int(m.group(1))] = float(m.group(2))
for n in np.argsort(-recur):
    print(f"neuron {n:2d}: readout condition {readout[n]:5.2f}; recurrent {recur[n]:6.2f}; first-order predicted 2% loss {pred[n]:.3f}; measured {meas[n]:.3f}")
for lab, v in (("readout condition", readout), ("recurrent condition", recur), ("first-order prediction", pred)):
    rho, p = spearmanr(v, meas); print(f"rank correlation with the measured 2% loss, {lab}: {rho:.2f} (p {p:.4f})")
tot = np.abs(T).sum((1, 2)) / lead
print(f"whole-network recurrent condition number, median over lists: {np.median(tot):.0f} (readout only: {np.median(np.abs(T[:, 9]).sum(1) / lead):.0f})")
