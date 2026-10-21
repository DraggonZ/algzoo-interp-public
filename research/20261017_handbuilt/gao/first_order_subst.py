"""Does the first-order (one backward pass) prediction forecast the cost of substituting a neuron's formula?
Predicted lead change: sum_t g_{t,n} (f_{t,n} - h_{t,n}), with g from the true run (recurrent_condition.py).
Measured: accuracy loss when the formula's values replace the neuron's at every step. 30-quantity menu formulas
(block_recurrent.plain). Single neurons, the four cancelling neurons together, all 16. 50,000 lists."""
import io, contextlib
import numpy as np
from scipy.stats import spearmanr
with contextlib.redirect_stdout(io.StringIO()):
    from block_recurrent import Xs, ys, plain
import simlib

X, y = Xs, ys; r = np.arange(len(X))
L, P, H = simlib.run(X); k = L.argmax(1); j = np.argsort(L, 1)[:, -2]; lead = L[r, k] - L[r, j]; acc0 = (k == y).mean()
g = simlib.WO[k] - simlib.WO[j]; Gt = np.zeros_like(H)
for t in range(9, -1, -1):
    Gt[:, t] = g; g = (g * (P[:, t] > 0)) @ simlib.WH
F = plain(list(range(16)))
def pred_acc(ns):
    d = sum((Gt[:, :, n] * (F[n] - H[:, :, n])).sum(1) for n in ns)
    new_k_is_y = np.where(lead + d < 0, j == y, k == y)   # first order: winner and runner-up swap if the lead turns negative
    return new_k_is_y.mean()
rows = []
for n in range(16):
    m = acc0 - (simlib.run(X, {n: F[n]})[0].argmax(1) == y).mean(); p = acc0 - pred_acc([n]); rows.append((m, p))
    print(f"neuron {n:2d}: measured loss {m:.3f}; first-order predicted {p:.3f}", flush=True)
R = np.array(rows); rho, pv = spearmanr(R[:, 0], R[:, 1])
print(f"single neurons: rank correlation {rho:.2f} (p {pv:.4f}); median |measured - predicted| {np.median(np.abs(R[:, 0] - R[:, 1])):.3f}")
for lab, ns in (("the four cancelling neurons", [1, 6, 7, 8]), ("all 16", list(range(16)))):
    m = acc0 - (simlib.run(X, {n: F[n] for n in ns})[0].argmax(1) == y).mean()
    print(f"{lab}: measured loss {m:.3f}; first-order predicted {acc0 - pred_acc(ns):.3f}")
