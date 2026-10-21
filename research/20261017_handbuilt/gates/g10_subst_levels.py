"""Substitution of the input-only gate rules (gaterules2.py) at three levels of detail for the history part D.
Fitted on 200,000 lists (seed 6), scored on 100,000 fresh lists (seed 8). Each gate alone, then both together.
Also reports how close the rule's values are to the trained neuron's on the intact run (rms over steps 1..9,
and the share of steps where the rule's on/off differs)."""
import pickle
import numpy as np
from gatelib import lists, run, acc_by_kind
from gaterules2 import fit, rule_values

KEYS = ["all", "S before L (revealed)", "S after L (kept)", "S at 8", "S at 9", "L at 8", "L at 9", "S right after L"]
X = lists(8, 100_000); L0, P0, H0 = run(X)
base = acc_by_kind(X, L0, KEYS)
print(" " * 34 + "  ".join(f"{k[:6]:>6s}" for k in KEYS))
print(f"{'intact':34s}" + "  ".join(f"{100 * base[k][0]:6.1f}" for k in KEYS))
Xf = lists(6)
for level in ("roles", "hinges", "gates"):
    co = fit(Xf, level)
    pickle.dump(co, open(f"g10_tables_{level}.pkl", "wb"))
    V, R, D = rule_values(X, co)
    for g in (5, 9):
        e = V[g][:, 1:] - H0[:, 1:, g]; sw = ((V[g][:, 1:] > 0) != (H0[:, 1:, g] > 0)).mean()
        print(f"  [{level}] neuron {g} rule vs trained on the intact run: rms {np.sqrt((e ** 2).mean()):.3f}, on/off differs on {100 * sw:.2f}% of steps")
    for name, sub in ((f"{level}: neuron 5", {5: V[5]}), (f"{level}: neuron 9", {9: V[9]}), (f"{level}: both", {5: V[5], 9: V[9]})):
        Lr = run(X, sub=sub)[0]; r = acc_by_kind(X, Lr, KEYS)
        print(f"{name:34s}" + "  ".join(f"{100 * r[k][0]:6.1f}" for k in KEYS) + f"   agree {100 * (Lr.argmax(1) == L0.argmax(1)).mean():.1f}%")
