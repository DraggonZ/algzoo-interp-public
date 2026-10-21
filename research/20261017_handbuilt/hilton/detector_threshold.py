"""Where does the trained detector (neuron 4) switch on, relative to the largest so far? Share of steps with neuron 4
on, binned by x / largest so far (largest above 0.3), split by how many steps ago the largest arrived. And the store
(neuron 2) against 17 x the largest, by steps since the largest arrived."""
import numpy as np
from algzoo.core import example
wi, wh, wo = (np.asarray(p, float) for p in example())
rng = np.random.default_rng(23); X = rng.standard_normal((400000, 10)); N = len(X)
h = np.zeros((N, 16)); HS = []
for t in range(10): h = np.maximum(h @ wh.T + X[:, t:t + 1] * wi, 0); HS.append(h)
HS = np.stack(HS, 1)
pm = np.maximum.accumulate(np.c_[np.zeros(N), np.maximum(X, 0)], 1)[:, :-1]     # largest before step t
rec = X > pm; last = np.full(N, -1); age = np.zeros((N, 10), int)
for t in range(10): age[:, t] = np.where(last >= 0, t - last, 99); last = np.where(rec[:, t], t, last)
q = X / np.maximum(pm, 1e-9); ok = (pm > 0.3) & (np.arange(10)[None, :] >= 1)
bins = [0.7, 0.8, 0.85, 0.9, 0.95, 1.0, 1.05]
print("share of steps with the detector on, by x / largest so far:")
for A in (1, 2, 4, 6):
    sel = ok & (age == A)
    print(f"   largest arrived {A} step(s) before: " + "  ".join(f"[{a:.2f},{b:.2f}) {(HS[:, :, 4][sel & (q >= a) & (q < b)] > 0).mean():.2f}" for a, b in zip(bins[:-1], bins[1:])))
print("store / largest so far, by steps since the largest arrived (median):")
m1now = np.maximum.accumulate(np.maximum(X, 0), 1)
for A in (1, 2, 3, 4, 6, 8):
    sel = (age == A) & (pm > 0.3)
    print(f"   {A}: {np.median(HS[:, :, 2][sel] / m1now[sel]):.2f}")
