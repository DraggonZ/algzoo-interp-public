"""Explain the two vote-placing events from the weights.

Event 1 (second arrives after the largest): the second subtracts r*|w| from each comparing neuron that is on. The score
change is r * (output weights of those neurons) @ (their input weights), a fixed vector per on/off set. Does it peak at
slot 9 (the newest)? How well does it match the actual change?
Event 2 (largest arrives after the second): split the scores at that step into parts from the ten other neurons, neuron 2,
neuron 4 and the rest. Which part puts the vote on the second's slot?
"""
import numpy as np

from algzoo.core import example

w_hi, w_hh, w_oh = example()
Wc = w_oh - w_oh.mean(0)
C = [1, 2, 4, 6, 7, 8]
Dn = [i for i in range(16) if i not in C]
np.set_printoptions(linewidth=200, precision=1, suppress=True)
h = np.maximum(0, w_hi); pre = [w_hi.copy()]; H = [h]
for k in range(1, 12):
    p = w_hh @ h; pre.append(p); h = np.maximum(0, p); H.append(h)

print("Event 1: predicted vote per unit of the second, by the largest's age when it arrives")
for g in range(1, 10):
    on = H[g - 1] @ w_hh.T > 0          # pre-activation at arrival (largest only) is pre[g]
    on = pre[g] > 0
    idx = [i for i in range(16) if on[i] and w_hi[i] < 0]
    a = Wc[:, idx] @ w_hi[idx]
    r = 0.7
    actual = Wc @ (np.maximum(0, pre[g] + r * w_hi) - np.maximum(0, pre[g]))
    print(f"  age {g}: neurons {idx}, predicted vote {r*a}, top slot {a.argmax()}, corr with actual {np.corrcoef(r*a, actual)[0,1]:.3f}")

print("\nEvent 2: largest = 1 arrives when the second (0.7) is a steps old; scores at that step split by neuron group")
r = 0.7
for a_ in range(1, 9):
    prev = r * H[a_ - 1]                 # second's state one step before the largest arrives
    hh = np.maximum(0, w_hh @ prev + 1.0 * w_hi)
    parts = {"ten other": Wc[:, Dn] @ hh[Dn], "n2": Wc[:, [2]] @ hh[[2]], "n4": Wc[:, [4]] @ hh[[4]],
             "n1,6,7,8": Wc[:, [1, 6, 7, 8]] @ hh[[1, 6, 7, 8]]}
    tot = sum(parts.values())
    slot = 9 - a_
    desc = ", ".join(f"{k} top {v.argmax()} (at slot {v[slot]:+.0f}, at 9 {v[9]:+.0f})" for k, v in parts.items())
    print(f"  age {a_}: right slot {slot}; total top {tot.argmax()} | {desc} | n4={hh[4]:.1f}, comps={np.round(hh[[1,6,7,8]],1)}")
