"""Is neuron 0's switching just 'off exactly two steps after a new largest'? Run the network with neuron 0's on/off
set by a rule (values from the weights), instead of by its own ReLU. Rules tried:
  age2      - off iff the largest arrived exactly two steps ago;
  age2+none - also off while no positive number has arrived, whenever its pre-activation is negative (true switch there).
Also: always on (no ReLU). 200,000 lists, seed 202."""
import numpy as np
from simlib import lists, run, score, task_vars, WH, WI
X = lists(202); L, P, _ = run(X); V = task_vars(X); age = V["age"]


def run_mask(rule):
    h = np.zeros((len(X), 16))
    for t in range(10):
        pre = h @ WH.T + X[:, t:t + 1] * WI; h = np.maximum(pre, 0)
        h[:, 0] = pre[:, 0] * rule(t, pre)
    return h @ WO.T


from simlib import WO
rules = {"always on": lambda t, pre: np.ones(len(X)),
         "age2": lambda t, pre: (age[:, t] != 2).astype(float),
         "age2+none": lambda t, pre: np.where(age[:, t] == -1, pre[:, 0] > 0, age[:, t] != 2).astype(float),
         "true ReLU": lambda t, pre: (pre[:, 0] > 0).astype(float)}
for k, r in rules.items():
    s = score(X, L, run_mask(r)); print(f"neuron 0 switched by {k:10s}: KL {s['kl']:.3f} acc {s['acc']:.3f} agree {s['agree']:.3f}")
