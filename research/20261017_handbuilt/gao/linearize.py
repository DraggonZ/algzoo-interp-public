"""Does a group's switching matter? Remove the ReLU from a group's neurons (they can go negative, never switch off),
the rest unchanged; also from single memory neurons. 200,000 lists, seed 202."""
import numpy as np
from simlib import GROUPS, WI, WH, WO, lists, run, score
X = lists(202); L = run(X)[0]


def run_lin(X, lin):
    h = np.zeros((len(X), 16)); m = np.zeros(16, bool); m[lin] = True
    for t in range(10):
        pre = h @ WH.T + X[:, t:t + 1] * WI; h = np.where(m, pre, np.maximum(pre, 0))
    return h @ WO.T


for name, ns in list(GROUPS.items()) + [("memory without 15", [0, 3, 10, 11, 12, 13, 14]), ("memory without 3, 15", [0, 10, 11, 12, 13, 14])] + [(f"neuron {n}", [n]) for n in GROUPS["memory"]]:
    s = score(X, L, run_lin(X, ns)); print(f"no ReLU in {name:22s}: KL {s['kl']:.3f} acc {s['acc']:.3f} agree {s['agree']:.3f}")

print("--- switching kept only in some memory neurons")
for keep in ([0, 15], [0], [15], [0, 14, 15], [0, 3, 15]):
    ns = [n for n in GROUPS["memory"] if n not in keep]
    s = score(X, L, run_lin(X, ns)); print(f"memory linear except {str(keep):12s}: KL {s['kl']:.3f} acc {s['acc']:.3f} agree {s['agree']:.3f}")
