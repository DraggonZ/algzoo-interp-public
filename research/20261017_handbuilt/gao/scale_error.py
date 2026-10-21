"""A nearly right account of the trained network: each neuron's true value, scaled by (1 + e) at every step and fed on
through the recurrence (a 'right idea, gain e off' error). One neuron at a time, and each role group. e = 2% and 10%.
Compared with the twin's store gain 2% off (58%, twin_graded_near.log). 100,000 lists."""
import numpy as np
import simlib
from simlib import lists

X = lists(808, 100_000); y = X.argsort(1)[:, -2]; base = (simlib.run(X)[0].argmax(1) == y).mean()


def scaled(ns, e):
    h = np.zeros((len(X), 16)); f = np.ones(16); f[ns] = 1 + e
    for t in range(10):
        h = np.maximum(h @ simlib.WH.T + X[:, t:t + 1] * simlib.WI, 0) * f
    return ((h @ simlib.WO.T).argmax(1) == y).mean()


print(f"trained accuracy {base:.3f}")
for e in (0.02, -0.02, 0.1):
    one = [scaled([n], e) for n in range(16)]
    print(f"gain {1 + e:.2f}, one neuron at a time: " + " ".join(f"{n}:{a:.3f}" for n, a in enumerate(one)) + f"; worst {min(one):.3f}")
    print("   groups: " + "; ".join(f"{g} {scaled(ns, e):.3f}" for g, ns in simlib.GROUPS.items()) + f"; all 16 {scaled(list(range(16)), e):.3f}", flush=True)
