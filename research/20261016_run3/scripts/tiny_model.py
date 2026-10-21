"""A tiny model of the two-spike case in score space, built from single-input runs and two weight products.

Predicted scores = m1 * L[age of largest] + c * m2 at the second's slot (+ variants), where L[a] = centered scores of a
lone input of size 1 at age a (10 x 10 table from ten single-input steps) and c = the newest-slot vote per unit input from
the weights (output weights of input-driven neurons times their input weights). No fitting to accuracies.
Agreement with the network on sampled two-spike inputs (top two of 10 Gaussians kept, rest zero), overall and on the
network's mistakes.
"""
import sys

import numpy as np

from algzoo.core import example, zoo

name = sys.argv[1] if len(sys.argv) > 1 else "example"
params = example() if name == "example" else zoo(16, 10, int(name))
w_hi, w_hh, w_oh = (np.asarray(p, np.float64) for p in params)
Wc = w_oh - w_oh.mean(0)
h = np.maximum(0, w_hi); H = [h]
for k in range(1, 10):
    h = np.maximum(0, w_hh @ h); H.append(h)
H = np.array(H)
L = H @ Wc.T                                   # [age, slot]
# newest-slot vote per unit input: change in scores when a smaller input (0.5) arrives on top of a lone input's state,
# averaged over the lone input's ages 2..8 (one step of the network on chosen states)
vs = []
for g in range(2, 9):
    pre = w_hh @ H[g - 1]
    vs.append(Wc @ (np.maximum(0, pre + 0.5 * w_hi) - np.maximum(0, pre)) / 0.5)
v = np.mean(vs, 0)
c = v[9] - np.delete(v, 9).mean()
print(f"model {name}: newest-slot vote per unit input c = {c:.1f}")

rng = np.random.default_rng(5)
X = rng.standard_normal((200000, 10))
o = np.argsort(X, 1)
p1, p2 = o[:, -1], o[:, -2]
m1, m2 = X[np.arange(len(X)), p1], X[np.arange(len(X)), p2]
X2 = np.zeros_like(X); X2[np.arange(len(X)), p1] = m1; X2[np.arange(len(X)), p2] = m2
hh = np.zeros((len(X), 16))
for t in range(10):
    hh = np.maximum(0, hh @ w_hh.T + X2[:, t, None] * w_hi)
net = (hh @ w_oh.T).argmax(1)
a1 = 9 - p1
S = m1[:, None] * L[a1]
S[np.arange(len(X)), p2] += c * np.maximum(m2, 0)
pred = S.argmax(1)
right = net == p2
print(f"network two-spike accuracy {right.mean():.4f}")
print(f"tiny model: same answer as network {np.mean(pred == net):.4f}; on network's mistakes, predicts the mistake {np.mean(pred[~right] != p2[~right]):.2f} "
      f"and the same wrong slot {np.mean(pred[~right] == net[~right]):.2f}; false alarms on network's right answers {np.mean(pred[right] != p2[right]):.4f}")
for lab, k in [("largest first", p1 < p2), ("second first", p1 > p2)]:
    print(f"  {lab}: agreement {np.mean(pred[k] == net[k]):.4f}, network accuracy {right[k].mean():.4f}")
