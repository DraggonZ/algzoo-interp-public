"""Back-to-back new largests in ARC: is neuron 9 (late gate) silenced when the next number is also a new largest?
Neuron 9 at step t+1 after a new largest at t, split by whether x_{t+1} is also a new largest. Also where its output
goes: weights from neuron 9 to the cancellers and to the memory neurons."""
import numpy as np
from algzoo.core import example

wi, wh, wo = (np.asarray(w, dtype=float) for w in example()); wi = wi.reshape(-1)
X = np.random.default_rng(31).standard_normal((300_000, 10)); B, n = X.shape
h = np.zeros((B, 16)); H = []
for t in range(n): h = np.maximum(h @ wh.T + X[:, t:t + 1] * wi, 0); H.append(h)
H = np.array(H)
z = np.zeros((B, 2)); m1p = np.stack([np.max(np.c_[z, X[:, :t]], 1) for t in range(n)], 1)
rec = (X > m1p) & (m1p > 0)                       # beat a positive largest
for t in range(1, 8):
    a = rec[:, t] & ~rec[:, t + 1]; b = rec[:, t] & rec[:, t + 1]
    if t == 3:
        print(f"t={t}: neuron 9 at t+1 after a new largest: next not a new largest {H[t + 1][a, 9].mean():.2f} (on {np.mean(H[t + 1][a, 9] > 0):.2f}); "
              f"next also a new largest {H[t + 1][b, 9].mean():.2f} (on {np.mean(H[t + 1][b, 9] > 0):.2f})")
allA = np.concatenate([H[t + 1][rec[:, t] & ~rec[:, t + 1], 9] for t in range(1, 8)])
allB = np.concatenate([H[t + 1][rec[:, t] & rec[:, t + 1], 9] for t in range(1, 8)])
print(f"all steps: next not a new largest: mean {allA.mean():.2f}, on {np.mean(allA > 0):.2f};  next also a new largest: mean {allB.mean():.2f}, on {np.mean(allB > 0):.2f}")
print("weights from neuron 9 to cancellers 1, 6, 7, 8:", np.round(wh[[1, 6, 7, 8], 9], 2))
print("weights from neuron 9 to memory 0, 3, 10-15:   ", np.round(wh[[0, 3, 10, 11, 12, 13, 14, 15], 9], 2))
print("weights from neuron 9 to gate 5 and to itself: ", np.round(wh[[5, 9], 9], 2))
