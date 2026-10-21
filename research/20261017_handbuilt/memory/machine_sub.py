"""Substitution test of the memory as a linear machine with switching rules.

The machine: m_t = D_t (A m_{t-1} + B o_{t-1} + w x_t), with A, B, w the trained weights into the memory (neurons
0, 3, 10-15) and D_t a 0/1 mask saying which memory neurons are on at step t. In the modal basis this is
z_t = Lambda z_{t-1} + V^-1 (writes), the five mode families of modes.log, with the masks as resets of single
coordinates. Without masks it is purely linear. The masks come from rules (below), not from the machine's own sign.

Three ways to connect it (o = the other eight neurons):
  closed : the other neurons run on their own weights and read the machine's memory; the machine reads them.
  hybrid : the machine reads the other neurons of the real (unchanged) run; the other neurons run on their own
           weights and read the machine's memory; scores from both.
  open   : the machine reads the real run's other neurons; scores = machine memory readout + real other neurons.
Rules (masks):
  linear          : all on.
  n0 age 2        : neuron 0 off exactly two steps after the largest so far arrived; all else on.
  age table       : each memory neuron off in the (step, age) cells where it is off on more than half the lists of a
                    different sample (seed 12); age = steps since the largest so far arrived, -1 = none positive yet.
  age table, 15 true : the age table, but neuron 15 switched by its own sign (to see what 15's rule must add).
  age table, 3+15 true: same, neurons 3 and 15 by their own sign.
  true ReLU       : every neuron by its own sign (must reproduce the trained network exactly).
Scored on 100,000 lists (seed 11) by kind of list."""
import numpy as np
from memlib import MEM, OTH, WI, WH, WO, run, task, lists, score, kinds, fmt

X = lists(11, 100_000); K = kinds(X); V = task(X); age = V["age"]
S0, P0, H0 = run(X); base = score(X, S0, S0, K)

# age table from another sample
Xc = lists(12, 100_000); Vc = task(Xc); _, Pc, _ = run(Xc)
table = np.zeros((10, 11, 16), bool)   # step, age+1, neuron -> off
for t in range(10):
    for a in range(-1, 10):
        m = Vc["age"][:, t] == a
        if m.sum() > 20: table[t, a + 1] = (Pc[m, t] <= 0).mean(0) > 0.5
tab_off = table[np.arange(10)[None, :], age + 1]   # (B, 10, 16)


def masks(rule):
    """returns function(t, pre_mem) -> (B, 8) on/off for the memory neurons"""
    if rule == "linear": return lambda t, pre: np.ones_like(pre, bool)
    if rule == "n0 age 2":
        def f(t, pre):
            on = np.ones_like(pre, bool); on[:, 0] = age[:, t] != 2; return on
        return f
    if rule.startswith("age table"):
        true = [] if rule == "age table" else ([7] if "3+15" not in rule else [1, 7])
        def f(t, pre):
            on = ~tab_off[:, t][:, MEM]
            for j in true: on[:, j] = pre[:, j] > 0
            return on
        return f
    if rule == "true ReLU": return lambda t, pre: pre > 0


def machine(rule, mode):
    mk = masks(rule); n = len(X); h = np.zeros((n, 16))
    for t in range(10):
        src = h if mode == "closed" else (H0[:, t - 1] if t else np.zeros((n, 16)))
        pre = h @ WH.T + X[:, t:t + 1] * WI                       # others read the machine's memory (closed, hybrid)
        pre_m = src[:, OTH] @ WH[np.ix_(MEM, OTH)].T + h[:, MEM] @ WH[np.ix_(MEM, MEM)].T + X[:, t:t + 1] * WI[MEM]
        new = np.maximum(pre, 0)
        new[:, MEM] = pre_m * mk(t, pre_m)
        if mode == "open": new[:, OTH] = H0[:, t][:, OTH]
        h = new
    return h @ WO.T


for rule in ["linear", "n0 age 2", "age table", "age table, 15 true", "age table, 3+15 true", "true ReLU"]:
    for mode in ["closed", "hybrid", "open"]:
        S = machine(rule, mode); r = score(X, S0, S, K)
        print(f"== rule {rule:22s} | {mode:6s}: acc {100 * r['all'][0]:.1f} ({100 * (r['all'][0] - base['all'][0]):+.1f}), agree {100 * r['all'][1]:.1f}, KL {r['all'][2]:.3f}")
        if mode == "closed" or rule == "age table": print(fmt(r, base))
print("trained:"); print(fmt(base))
print("age table: cells where each neuron is off (step: ages)")
for j, nn in enumerate(MEM):
    cells = [f"{t}:" + ",".join(str(a - 1) for a in range(11) if table[t, a, nn]) for t in range(10) if table[t, :, nn].any()]
    print(f"  neuron {nn:2d}: " + "  ".join(cells))
