"""Per-neuron condition number: does a neuron's push on the winner's lead, relative to the lead, predict how much a
small error in that neuron costs? Per neuron: median over lists of |(w_o[winner] - w_o[runner-up])_n h_n| / lead.
Compared with the accuracy lost when that neuron's true values are scaled by 1.02 or 1.05 at every step (fed through
the recurrence), and with noise of 1% on that neuron alone. Trained network, 100,000 lists."""
import numpy as np
from scipy.stats import spearmanr
import simlib

X = simlib.lists(2222, 100_000); y = X.argsort(1)[:, -2]; r = np.arange(len(X))
L, _, H = simlib.run(X); k = L.argmax(1); j = np.argsort(L, 1)[:, -2]; lead = L[r, k] - L[r, j]; acc0 = (k == y).mean()
cond = np.median(np.abs((simlib.WO[k] - simlib.WO[j]) * H[:, 9]) / np.maximum(lead, 1e-12)[:, None], 0)
typ = np.array([np.median(H[:, :, n][H[:, :, n] > 0]) for n in range(16)]); rng = np.random.default_rng(0)


def loss(n, scale=1.0, noise=0.0):
    h = np.zeros((len(X), 16)); f = np.ones(16); f[n] = scale; g = np.zeros(16); g[n] = noise
    for t in range(10):
        h = np.maximum(h @ simlib.WH.T + X[:, t:t + 1] * simlib.WI, 0) * f
        h = np.maximum(h + g * typ * rng.standard_normal(h.shape) * (h > 0), 0)
    return acc0 - ((h @ simlib.WO.T).argmax(1) == y).mean()


rows = [(n, cond[n], loss(n, 1.02), loss(n, 1.05), loss(n, noise=0.03)) for n in range(16)]
for n, c, a, b, d in sorted(rows, key=lambda x: -x[1]):
    print(f"neuron {n:2d}: push on the lead / lead (median) {c:5.2f}; loss with values 2% too large {a:.3f}, 5% {b:.3f}, 3% noise {d:.3f}")
R = np.array([r[1:] for r in rows])
for i, lab in ((1, "2% scale"), (2, "5% scale"), (3, "3% noise")):
    rho, p = spearmanr(R[:, 0], R[:, i]); print(f"rank correlation, per-neuron condition vs loss with {lab}: {rho:.2f} (p {p:.4f})")
