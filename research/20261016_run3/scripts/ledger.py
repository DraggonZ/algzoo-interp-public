"""Vote ledger on full random inputs: does 'every input except the largest leaves a vote at its slot' predict the network?

First, measure from the network on random inputs: the final centered scores against each input's value at its own slot
(how big is the vote per unit value, by age, for non-largest inputs?). Then test ledger models:
  V0: score_k = x_k for every k except the largest (excluded)  -> the ideal rule (always right)
  V1: score_k = lam[age_k] * x_k, lam measured from the two-spike runs (no fitting to accuracy)
  V2: V1 plus the largest's leftover table L[age] * x_max (from single-input runs)
Agreement with the network's answer on 200,000 random inputs.
"""
import sys

import numpy as np

from algzoo.core import example, zoo

name = sys.argv[1] if len(sys.argv) > 1 else "example"
params = example() if name == "example" else zoo(16, 10, int(name))
w_hi, w_hh, w_oh = (np.asarray(p, np.float64) for p in params)
Wc = w_oh - w_oh.mean(0)

# two-spike vote height per unit of the second, by its age (largest = 1 first, second = 0.5 later), averaged over gaps
lam = np.zeros(10); cnt = np.zeros(10)
for p1 in range(10):
    for p2 in range(p1 + 1, 10):
        def fin(x):
            h = np.zeros(16)
            for t in range(10):
                h = np.maximum(0, w_hh @ h + x[t] * w_hi)
            return Wc @ h
        x1 = np.zeros(10); x1[p1] = 1.0
        x2 = x1.copy(); x2[p2] = 0.5
        d = (fin(x2) - fin(x1)) / 0.5
        lam[9 - p2] += d[p2] - np.delete(d, p2).mean(); cnt[9 - p2] += 1
lam /= np.maximum(cnt, 1)
print(f"model {name}: vote height per unit value by age 0..8:", np.round(lam[:9], 1))

h = np.maximum(0, w_hi); H = [h]
for k in range(1, 10):
    h = np.maximum(0, w_hh @ h); H.append(h)
L = np.array(H) @ Wc.T

rng = np.random.default_rng(7)
X = rng.standard_normal((200000, 10))
hh = np.zeros((len(X), 16))
for t in range(10):
    hh = np.maximum(0, hh @ w_hh.T + X[:, t, None] * w_hi)
S = hh @ Wc.T
net = S.argmax(1)
o = np.argsort(X, 1); pmax, p2nd = o[:, -1], o[:, -2]
print(f"network accuracy {np.mean(net == p2nd):.4f}")
ages = 9 - np.arange(10)
# measured: regression of final score at slot k on x_k for non-largest inputs, by age
for a in (0, 2, 5, 8):
    k = 9 - a
    m = pmax != k
    sl = np.polyfit(X[m, k], S[m, k] - S[m].mean(1), 1)[0]
    print(f"  measured slope of score at slot {k} (age {a}) on its own value: {sl:.1f}")
V0 = X.copy(); V0[np.arange(len(X)), pmax] = -np.inf
V1 = X * np.r_[lam[::-1][1:], lam[0]][None, :] if False else X * lam[ages][None, :]
V1[np.arange(len(X)), pmax] = -np.inf
V2 = X * lam[ages][None, :] + X[np.arange(len(X)), pmax][:, None] * L[ages[pmax]]
V2[np.arange(len(X)), pmax] = -np.inf
for lab, V in [("V0 ideal rule", V0), ("V1 ledger with vote heights", V1), ("V2 ledger + largest's leftover", V2)]:
    pr = V.argmax(1)
    wrong = net != p2nd
    print(f"{lab}: same answer as network {np.mean(pr == net):.4f}; predicts network's mistakes {np.mean(pr[wrong] != p2nd[wrong]):.2f}, "
          f"same wrong slot {np.mean(pr[wrong] == net[wrong]):.2f}; false alarms {np.mean(pr[~wrong] != p2nd[~wrong]):.4f}")
