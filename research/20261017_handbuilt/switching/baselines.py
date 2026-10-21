"""Step 1: baselines on fresh lists (seed 31, 200,000), fits on seed 32: no switching; neuron 0 off at age 2; the
step-by-age majority table; each neuron by its own sign (must reproduce the network)."""
import numpy as np
from swlib import *

X = lists(TEST_SEED, 200_000); K = kinds(X); V = task(X); age = V["age"]
S0, P0, H0 = run(X); base = score(X, S0, S0, K); true_off = P0[:, :, MEM] <= 0
print(f"network: {100 * base['all'][0]:.2f}")
Xf = lists(FIT_SEED, 200_000); Vf = task(Xf); _, Pf, _ = run(Xf)
table = np.zeros((10, 11, 8), bool)
for t in range(10):
    for a in range(-1, t + 1):
        m = Vf["age"][:, t] == a
        if m.sum() > 20: table[t, a + 1] = (Pf[m, t][:, MEM] <= 0).mean(0) > 0.5
print("fitted on/off cells in the table:", sum(((Vf['age'][:, t] == a).sum() > 20) for t in range(10) for a in range(-1, t + 1)) * 8)
rules = {
    "no switching": lambda t, c: np.zeros((len(X), 8), bool),
    "neuron 0 off at age 2, rest on": lambda t, c: np.c_[age[:, t] == 2, np.zeros((len(X), 7), bool)],
    "step-by-age table": lambda t, c: table[t, age[:, t] + 1],
    "own sign": lambda t, c: c["pre_m"] <= 0,
}
for name, f in rules.items():
    S, offs = machine(X, f); report(name, X, S0, S, offs, true_off, base, K if name != "own sign" else None)
