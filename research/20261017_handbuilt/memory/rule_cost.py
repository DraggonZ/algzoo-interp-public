"""Question: which memory neurons' switching does the age table (machine_sub.py) get wrong in a way that matters?
Start from the trained network (every neuron by its own sign) and switch one memory neuron (or a set) by the age
table instead; closed loop. Also the clipped amount: mean of max(-pre-activation, 0) by age class, for each neuron,
and the share of steps where the table and the true sign disagree. 100,000 lists, seed 11; table from seed 12."""
import numpy as np
from memlib import MEM, run, task, lists, score, kinds, fmt

X = lists(11, 100_000); K = kinds(X); V = task(X); age = V["age"]
S0, P0, H0 = run(X); base = score(X, S0, S0, K)
Xc = lists(12, 100_000); Vc = task(Xc); _, Pc, _ = run(Xc)
table = np.zeros((10, 11, 16), bool)
for t in range(10):
    for a in range(-1, 10):
        m = Vc["age"][:, t] == a
        if m.sum() > 20: table[t, a + 1] = (Pc[m, t] <= 0).mean(0) > 0.5
tab_off = table[np.arange(10)[None, :], age + 1]


def by_table(ns):
    def hook(t, pre, h):
        h = h.copy()
        for n in ns: h[:, n] = np.where(tab_off[:, t, n], 0.0, pre[:, n])
        return h
    return hook


cls = {"none yet": age == -1, "age 0": age == 0, "age 1": age == 1, "age 2": age == 2, "age 3": age == 3, "age 4": age == 4, "age 5": age == 5, "age 6": age == 6, "age 7+": age >= 7}
print("clipped amount (mean of max(-pre, 0)) and table-vs-sign disagreement, by age class")
for n in MEM:
    clip = np.maximum(-P0[:, :, n], 0); dis = tab_off[:, :, n] != (P0[:, :, n] <= 0)
    print(f"  neuron {n:2d} clip: " + " ".join(f"{k} {clip[m].mean():5.2f}" for k, m in cls.items()))
    print(f"            disagree: " + " ".join(f"{k} {dis[m].mean():5.2f}" for k, m in cls.items()))
print("\naccuracy when the listed neurons switch by the age table and all others by their own sign (closed loop)")
for ns in [[n] for n in MEM] + [[0, 10, 11, 12, 13, 14], [10, 11, 12, 13, 14], [3, 15], MEM]:
    S = run(X, by_table(ns))[0]; r = score(X, S0, S, K)
    print(f"  {str(ns):28s} acc {100 * r['all'][0]:.1f} ({100 * (r['all'][0] - base['all'][0]):+.1f})  " +
          " ".join(f"{k[:14]} {100 * (r[k][0] - base[k][0]):+.1f}" for k in ["new largest at step 9", "new largest at step 8", "two in a row, earlier", "ordinary"]))
