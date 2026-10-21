"""r1b diagnostic: one-step least squares (ARC's previous hiding values, blind feedback, on steps only) of each hiding
neuron's input minus w_in x, for several feature sets. Prints rms residual per neuron. Cheap guide, not the score."""
import sys
sys.path.insert(0, "rules")
import numpy as np
from common import HID, WI, account_features, sd_rule, task_vars, lag
import simlib

X = simlib.lists(101, 50_000); _, P, H = simlib.run(X)
V = task_vars(X); sd = sd_rule(X)
Hp = np.concatenate([np.zeros((len(X), 1, 16)), H[:, :-1]], 1)
blind = [Hp[..., k] - lag(X) * WI[k] for k in HID]
base = {"m1": V["m1"], "m2": V["m2"], "store": sd[..., 0], "det": sd[..., 1], "jump1": lag(V["jump"]),
        "jump2": lag(V["jump"], 2), "nl2": lag(V["newmax"] * V["x"], 2), "m2in1": lag(V["m2in"]), "m1in1": lag(V["m1in"]),
        "xprev": lag(X), "rec1": lag(V["newmax"])}
SETS = {
    "lean (no waves)": ["m1", "m2", "store", "det", "jump1", "jump2", "nl2"],
    "- m1": ["m2", "store", "det", "jump1", "jump2", "nl2"],
    "- jump1": ["m1", "m2", "store", "det", "jump2", "nl2"],
    "- store": ["m1", "m2", "det", "jump1", "jump2", "nl2"],
    "- det": ["m1", "m2", "store", "jump1", "jump2", "nl2"],
    "- m2": ["m1", "store", "det", "jump1", "jump2", "nl2"],
    "- jump2": ["m1", "m2", "store", "det", "jump1", "nl2"],
    "- nl2": ["m1", "m2", "store", "det", "jump1", "jump2"],
    "no blindness (+xprev)": ["m1", "m2", "store", "det", "jump1", "jump2", "nl2", "xprev"],
}
for name, fs in SETS.items():
    res = []
    for i, n in enumerate(HID):
        A = np.stack([base[f] for f in fs] + blind, -1).reshape(-1, len(fs) + 4)
        y = (P[:, :, n] - WI[n] * X).reshape(-1); on = (P[:, :, n] > 0).reshape(-1)
        c = np.linalg.lstsq(A[on], y[on], rcond=None)[0]
        res.append(np.sqrt(((A[on] @ c - y[on]) ** 2).mean()))
    print(f"{name:24s} " + "  ".join(f"{r:.3f}" for r in res))
