"""r1b diagnostic: for the store-reference feature set (r1b_store_ref.py), the one-step rms residual of each hiding
neuron when each single feature is dropped for that neuron only (ARC's previous hiding values, blind feedback, on steps)."""
import sys
sys.path.insert(0, "rules")
import numpy as np
from common import HID, WI, sd_rule, task_vars, lag
import simlib

X = simlib.lists(101, 50_000); _, P, H = simlib.run(X)
V = task_vars(X); sd = sd_rule(X)
Hp = np.concatenate([np.zeros((len(X), 1, 16)), H[:, :-1]], 1)
blind = [Hp[..., k] - lag(X) * WI[k] for k in HID]
base = {"m2": V["m2"], "store": sd[..., 0], "det": sd[..., 1], "jump1": lag(V["jump"]), "jump2": lag(V["jump"], 2),
        "nl2": lag(V["newmax"] * V["x"], 2)}
for n in HID:
    y = (P[:, :, n] - WI[n] * X).reshape(-1); on = (P[:, :, n] > 0).reshape(-1)
    out = []
    for drop in [None] + list(base):
        fs = [f for f in base if f != drop]
        A = np.stack([base[f] for f in fs] + blind, -1).reshape(-1, len(fs) + 4)[on]
        c = np.linalg.lstsq(A, y[on], rcond=None)[0]
        out.append(f"{'none' if drop is None else '-' + drop} {np.sqrt(((A @ c - y[on]) ** 2).mean()):.3f}")
    print(f"neuron {n}: " + ", ".join(out))
