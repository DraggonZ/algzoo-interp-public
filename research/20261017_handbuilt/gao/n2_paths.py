"""Where is neuron 2's timing used? Neuron 2's values from lists with the largest moved to a random earlier place
(store claim with the second-largest kept, gao/scrub.py), substituted (a) only in the readout (final scores), (b) only in
what other neurons receive at each step, (c) both. Same with the place kept, as the reference. 100,000 lists."""
import numpy as np
import scrub
from simlib import WI, WH, WO, lists, run, score
X = lists(202, 100_000); L = run(X)[0]; y = X.argsort(1)[:, -2]


def values(opts):
    V = np.zeros((len(X), 10))
    for t in range(10):
        V[:, t] = scrub.state_at(scrub.claim_list(X, t, "store+detector", opts), t)[:, 2]
    return V


def run_paths(V, readout, recurrent):
    h = np.zeros((len(X), 16))
    for t in range(10):
        hin = h.copy()
        if recurrent and t > 0:
            hin[:, 2] = V[:, t - 1]
        h = np.maximum(hin @ WH.T + X[:, t:t + 1] * WI, 0)
    hout = h.copy()
    if readout:
        hout[:, 2] = V[:, 9]
    return hout @ WO.T


for lab, opts in (("largest moved", ("m2",)), ("place kept", ("m1pos", "m2"))):
    V = values(opts)
    for name, ro, rc in (("readout only", True, False), ("other neurons only", False, True), ("both", True, True)):
        s = score(X, L, run_paths(V, ro, rc)); print(f"{lab:13s} {name:18s}: acc {s['acc']:.3f} agree {s['agree']:.3f} KL {s['kl']:.3f}", flush=True)

print("--- neuron 2's timing sent only to some receivers (largest moved)")
V = values(("m2",))
def run_recv(V, receivers):
    h = np.zeros((len(X), 16))
    for t in range(10):
        pre = h @ WH.T + X[:, t:t + 1] * WI
        if t > 0:   # replace neuron 2's contribution for the chosen receivers
            pre[:, receivers] += np.outer(V[:, t - 1] - h[:, 2], WH[receivers, 2])
        h = np.maximum(pre, 0)
    return h @ WO.T
for name, rec in (("cancellers 1, 6, 7, 8", [1, 6, 7, 8]), ("detector 4", [4]), ("store 2 itself", [2]), ("memory", [0, 3, 10, 11, 12, 13, 14, 15]), ("gates 5, 9", [5, 9])):
    s = score(X, L, run_recv(V, rec)); print(f"to {name:22s}: acc {s['acc']:.3f} agree {s['agree']:.3f}", flush=True)
