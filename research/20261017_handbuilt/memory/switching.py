"""Question: where does the memory's switching matter? For each memory neuron, (1) how often it is off, by the age of
the largest so far (steps since it arrived; -1 = no positive number yet); (2) the accuracy cost of removing its ReLU
only at steps of one age class (it may then go negative there), keeping the true ReLU elsewhere and in all other
neurons. Closed loop: the rest of the network runs on its own weights and reads the changed memory.
100,000 lists, seed 11."""
import numpy as np
from memlib import MEM, run, task, lists, score, kinds

X = lists(11, 100_000); V = task(X); age = V["age"]; S0, P0, H0 = run(X); base = score(X, S0, S0)["all"][0]
classes = {"none yet": age == -1, "age 0": age == 0, "age 1": age == 1, "age 2": age == 2, "age 3": age == 3, "age 4-5": (age >= 4) & (age <= 5), "age 6+": age >= 6}
print(f"trained accuracy {100 * base:.1f}")
print("(1) off-rate (pre-activation <= 0) by age class; share of steps in each class: " + " ".join(f"{k} {m.mean():.2f}" for k, m in classes.items()))
for n in MEM:
    off = P0[:, :, n] <= 0
    print(f"  neuron {n:2d}: all {off.mean():.2f} | " + " ".join(f"{k} {off[m].mean():.2f}" for k, m in classes.items()))


def lin_where(neurons, mask):
    def hook(t, pre, h):
        h = h.copy()
        for n in neurons: h[:, n] = np.where(mask[:, t], pre[:, n], h[:, n])
        return h
    return hook


print("(2) accuracy change (points) when the ReLU is removed only in one age class")
print("  neuron   " + " ".join(f"{k:>9s}" for k in classes) + "    all steps")
for ns in [[n] for n in MEM] + [[3, 10, 11, 12, 13, 14], MEM]:
    row = []
    for k, m in classes.items():
        S = run(X, lin_where(ns, m))[0]; row.append(100 * (score(X, S0, S)["all"][0] - base))
    S = run(X, lin_where(ns, np.ones_like(age, bool)))[0]; row.append(100 * (score(X, S0, S)["all"][0] - base))
    print(f"  {str(ns):22s}" if len(ns) > 1 else f"  {ns[0]:6d}   ", " ".join(f"{x:9.1f}" for x in row))
