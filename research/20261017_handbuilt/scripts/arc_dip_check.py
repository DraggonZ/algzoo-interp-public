"""Does ARC use v3's dip? On the step after a new largest, neuron 4 (the detector) fires in proportion to the
excess. Its outgoing weights to the timer neurons, and the actual change in timer neurons' pre-activation on the
step after a record versus a non-record, on random lists."""
import numpy as np
from algzoo.core import example

wi, wh, wo = (np.asarray(w, dtype=float) for w in example()); wi = wi.reshape(-1)
canc = [1, 6, 7, 8]; tim = [i for i in range(16) if i not in canc + [2, 4, 5]]
np.set_printoptions(precision=2, suppress=True, linewidth=150)
print("timer neurons", tim)
print("weights from neuron 4 (detector) to timer neurons:", wh[tim, 4])
print("weights from neuron 4 to cancellers 1, 6, 7, 8:   ", wh[canc, 4])
print("weights from neuron 2 (store) to timer neurons:   ", wh[tim, 2])
print("weights from neuron 5 (gate) to timer neurons:    ", wh[tim, 5])
print("input weights of timer neurons:                   ", wi[tim])
rng = np.random.default_rng(12)
X = rng.standard_normal((200_000, 10))
h = np.zeros((len(X), 16)); fr_after_rec, fr_other = [], []
m1 = np.zeros(len(X)); rec_prev = np.zeros(len(X), bool)
for t in range(10):
    pre = h @ wh.T + X[:, t:t + 1] * wi
    if t >= 2:
        on = pre[:, tim] > 0
        fr_after_rec.append(on[rec_prev].mean(0)); fr_other.append(on[~rec_prev].mean(0))
    rec_prev = (X[:, t] > m1) & (m1 > 0); m1 = np.maximum(m1, X[:, t])
    h = np.maximum(pre, 0)
print("share of timer neurons on, step after a record that beat a positive largest:", np.mean(fr_after_rec, 0))
print("share on, other steps:                                                    ", np.mean(fr_other, 0))
