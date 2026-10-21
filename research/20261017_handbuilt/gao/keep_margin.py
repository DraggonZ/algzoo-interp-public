"""Does forgetting widen ARC's margins? Force the gate (neuron 5) to keep forgotten numbers (as gate_keep_all.py:
neuron 5 set to s x 11.51 x the number one step later, s = 0.25 and 0.5), then measure accuracy, the median ratio of
the decision margin to the pushes on the winner's lead, the median margin, and accuracy with 1% noise on every neuron.
100,000 lists."""
import numpy as np
import simlib
from simlib import lists, task_vars

X = lists(1515, 100_000); y = X.argsort(1)[:, -2]; B = len(X); V = task_vars(X)
forg = (V["x"] > 0) & ~(V["newmax"] > 0) & (V["x"] < 0.75 * V["m2"])
_, _, H0 = simlib.run(X); typ = np.array([np.median(H0[:, :, n][H0[:, :, n] > 0]) for n in range(16)])
rng = np.random.default_rng(0)


def run(s, noise=0.0):
    h = np.zeros((B, 16))
    for t in range(10):
        h = np.maximum(h @ simlib.WH.T + X[:, t:t + 1] * simlib.WI, 0)
        if t and s:
            f = forg[:, t - 1]; h[f, 5] = s * 11.51 * X[f, t - 1]
        if noise:
            h = np.maximum(h + noise * typ * rng.standard_normal(h.shape) * (h > 0), 0)
    return h


for s in (0.0, 0.25, 0.5):
    h = run(s); L = h @ simlib.WO.T; k = L.argmax(1); j = np.argsort(L, 1)[:, -2]
    lead = L[np.arange(B), k] - L[np.arange(B), j]; push = np.abs((simlib.WO[k] - simlib.WO[j]) * h).sum(1)
    acc_n = ((run(s, 0.01) @ simlib.WO.T).argmax(1) == y).mean()
    print(f"gate keeps forgotten numbers at share {s}: accuracy {(k == y).mean():.3f}; median margin {np.median(lead):.2f}; "
          f"margin/pushes {np.median(lead / push):.4f}; accuracy with 1% noise {acc_n:.3f}", flush=True)
