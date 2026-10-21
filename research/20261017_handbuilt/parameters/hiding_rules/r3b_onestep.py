"""r3b diagnostic: one-step fits on ARC's states (blind feedback as in the rules) for feature sets that read the
network's own neurons one step back: store (2), detector (4), late gate (9), early gate (5), against the list
quantities of r2b_shared_store. Prints rms per hiding neuron (on steps) and the fitted coefficients."""
import sys
sys.path.insert(0, "rules")
import numpy as np
from common import HID, WI, task_vars, lag
import simlib

X = simlib.lists(101, 50_000); _, P, H = simlib.run(X)
V = task_vars(X)
Hp = np.concatenate([np.zeros((len(X), 1, 16)), H[:, :-1]], 1)
blind = Hp[..., HID] - lag(X)[..., None] * WI[HID]
F = {"m2": V["m2"], "store": Hp[..., 2], "det": Hp[..., 4], "pjump": lag(V["jump"]), "jump2": lag(V["jump"], 2),
     "nm2": lag(V["newmax"] * V["x"], 2), "gate9": Hp[..., 9], "gate5": Hp[..., 5]}
SETS = {
    "r2b features":        [["m2", "store", "det", "pjump", "jump2", "nm2"]] * 2 + [["m2", "store", "det"], ["m2", "store", "det", "pjump", "jump2"]],
    "no prev jump":        [["m2", "store", "det", "jump2", "nm2"]] * 2 + [["m2", "store", "det"], ["m2", "store", "det", "jump2"]],
    "gate9 for 2-back":    [["m2", "store", "det", "pjump", "gate9"]] * 2 + [["m2", "store", "det"], ["m2", "store", "det", "pjump", "gate9"]],
    "gate9, no prev jump": [["m2", "store", "det", "gate9"]] * 2 + [["m2", "store", "det"], ["m2", "store", "det", "gate9"]],
    "gate9 all four":      [["m2", "store", "det", "gate9"]] * 4,
    "gate9+5 all four":    [["m2", "store", "det", "gate9", "gate5"]] * 4,
}
np.set_printoptions(precision=3, suppress=True, linewidth=160)
for name, sets in SETS.items():
    rs, cs = [], []
    for i, n in enumerate(HID):
        A = np.concatenate([np.stack([F[f] for f in sets[i]], -1), blind], -1).reshape(-1, len(sets[i]) + 4)
        y = (P[:, :, n] - WI[n] * X).reshape(-1); on = (P[:, :, n] > 0).reshape(-1)
        c = np.linalg.lstsq(A[on], y[on], rcond=None)[0]
        rs.append(np.sqrt(((A[on] @ c - y[on]) ** 2).mean())); cs.append(dict(zip(sets[i], c[:len(sets[i])].round(3))))
    print(f"{name:22s} rms {np.array(rs)}  n const {sum(len(s) for s in sets)}")
    for i, n in enumerate(HID):
        print(f"      {n}: {cs[i]}")
