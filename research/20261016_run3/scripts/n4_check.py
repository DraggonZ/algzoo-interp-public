"""Does neuron 4 fire in proportion to how much a new input exceeds the running largest? (full random inputs)"""
import numpy as np

from algzoo.core import example

w_hi, w_hh, w_oh = (np.asarray(p, np.float64) for p in example())
rng = np.random.default_rng(4)
X = rng.standard_normal((100000, 10))
h = np.zeros((len(X), 16))
xs, gaps, n4 = [], [], []
for t in range(10):
    h = np.maximum(0, h @ w_hh.T + X[:, t, None] * w_hi)
    if t >= 1:
        M = X[:, :t].max(1)
        gaps.append(X[:, t] - M); n4.append(h[:, 4])
g = np.concatenate(gaps); v = np.concatenate(n4)
pos = g > 0
print(f"new largest: corr(neuron 4, excess over running largest) {np.corrcoef(v[pos], g[pos])[0,1]:.3f}; slope {np.polyfit(g[pos], v[pos], 1)[0]:.2f} per unit (input weight 10.16)")
print(f"not a new largest: neuron 4 is zero in {np.mean(v[~pos] < 1e-6):.3f} of steps; mean {v[~pos].mean():.3f}")
