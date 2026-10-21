"""Where is neuron 8's knowledge of the second-largest's place used? Neuron 8's values from lists that keep the input,
the largest (value and place) and the second-largest's value but move the second-largest to a random place (the
cancellers' base claim, gao/scrub.py), substituted (a) in the readout only, (b) for each group of receivers only.
Reference: the same with the second-largest's place kept. 100,000 lists."""
import numpy as np
import scrub
from simlib import WI, WH, WO, lists, run, score
X = lists(202, 100_000); L = run(X)[0]


def values(opts):
    V = np.zeros((len(X), 10))
    for t in range(10):
        V[:, t] = scrub.state_at(scrub.claim_list(X, t, "cancellers", opts), t)[:, 8]
    return V


def run_paths(V, receivers, readout):
    h = np.zeros((len(X), 16))
    for t in range(10):
        pre = h @ WH.T + X[:, t:t + 1] * WI
        if t > 0 and receivers:
            pre[:, receivers] += np.outer(V[:, t - 1] - h[:, 8], WH[receivers, 8])
        h = np.maximum(pre, 0)
    hout = h.copy()
    if readout:
        hout[:, 8] = V[:, 9]
    return hout @ WO.T


for lab, opts in (("second moved", ()), ("place kept", ("m2pos",))):
    V = values(opts)
    for name, rec, ro in (("readout only", [], True), ("cancellers 1, 6, 7", [1, 6, 7], False), ("itself", [8], False),
                          ("gates 5, 9", [5, 9], False), ("store and detector", [2, 4], False),
                          ("memory", [0, 3, 10, 11, 12, 13, 14, 15], False), ("everything", list(range(16)), True)):
        s = score(X, L, run_paths(V, rec, ro)); print(f"{lab:12s} {name:20s}: acc {s['acc']:.3f} agree {s['agree']:.3f}", flush=True)
