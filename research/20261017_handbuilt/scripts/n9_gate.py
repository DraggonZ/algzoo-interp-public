"""Is ARC's neuron 9 a one-step-late new-largest gate? On random lists: neuron 9's activity at step t against
whether step t-1 brought a new largest (beating a positive largest, or the first positive), and against that
number's value and excess. Also its weights from the store, detector and cancellers."""
import numpy as np
from algzoo.core import example

wi, wh, wo = (np.asarray(w, dtype=float) for w in example()); wi = wi.reshape(-1)
X = np.random.default_rng(29).standard_normal((300_000, 10)); B, n = X.shape
h = np.zeros((B, 16)); H = []
for t in range(n): h = np.maximum(h @ wh.T + X[:, t:t + 1] * wi, 0); H.append(h)
H = np.array(H)
z = np.zeros((B, 2)); m1p = np.stack([np.max(np.c_[z, X[:, :t]], 1) for t in range(n)], 1)
rec = X > m1p
n9 = H[1:, :, 9].T                     # neuron 9 at steps 1..9
prev_rec = rec[:, :-1]; xprev = X[:, :-1]; excess = np.maximum(X[:, :-1] - m1p[:, :-1], 0)
print(f"neuron 9 on: after a new largest {np.mean(n9[prev_rec] > 0):.2f}, otherwise {np.mean(n9[~prev_rec] > 0):.2f}")
print(f"mean activity: after a new largest {n9[prev_rec].mean():.2f}, otherwise {n9[~prev_rec].mean():.2f}")
for name, v in (("new largest's value", xprev[prev_rec]), ("its excess over the old largest", excess[prev_rec]), ("old largest", m1p[:, :-1][prev_rec])):
    print(f"  correlation of neuron 9 (after a new largest) with {name}: {np.corrcoef(n9[prev_rec], v)[0, 1]:.2f}")
print("weights into neuron 9 from store 2, detector 4, gate 5, cancellers 1, 6, 7, 8, input:",
      np.round(wh[9, [2, 4, 5, 1, 6, 7, 8]], 2), round(wi[9], 2))
print("neuron 9's weights out to the other timer neurons 0, 3, 10-15:", np.round(wh[[0, 3, 10, 11, 12, 13, 14, 15], 9], 2))
