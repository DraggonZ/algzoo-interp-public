"""How well do coefficient tables by role and age describe the parts of each gate's pre-activation?
Parts (g05_structure.py): the switch points r_j of detector 4 and hiding neurons 1, 6, 7, 8 at step t, and the rest
D_g(t) (memory, store, gates' own pushes, next input). All depend only on the list before x_t (plus x_{t+1}, the gate's
own previous value, and a tiny slope in x_t for D).
Least squares on 200,000 lists (seed 6), scored on 200,000 others (seed 7), steps t = 1..8 pooled.
Feature sets: 'basic' = x_t, m1, m2 (no ages); 'roles' = rolefeat.role_features (value x role x age indicators)."""
import numpy as np
from gatelib import WI, WH, lists, run
from rolefeat import role_features

HJ = [1, 4, 6, 7, 8]


def parts(X):
    L, P, H = run(X)
    out = {}
    for t in range(1, 9):
        u = H[:, t - 1] @ WH.T
        d = {f"r{j}": -u[:, j] / WI[j] for j in HJ}
        for g in (5, 9):
            d[f"D{g}"] = P[:, t + 1, g] - sum(WH[g, j] * H[:, t, j] for j in HJ)
        d["g5prev"] = H[:, t, 5]; d["g9prev"] = H[:, t, 9]
        out[t] = d
    keep = np.zeros((len(X), 10), bool); keep[:, :9] = H[:, 1:, 5] > 0
    return out, keep, H


def design(X, t, keep, d, kind, extra):
    B = len(X)
    if kind == "basic":
        from gatelib import task_vars
        V = task_vars(X)
        F = np.c_[V["m1"][:, t], V["m2"][:, t]]
    else:
        F = role_features(X, t, keep)[0]
    cols = [F]
    if "x" in extra: cols.append(X[:, t:t + 1])
    if "next" in extra: cols.append(X[:, t + 1:t + 2])
    if "g5prev" in extra: cols.append(d["g5prev"][:, None])
    if "g9prev" in extra: cols.append(d["g9prev"][:, None])
    return np.concatenate(cols, 1)


Xa, Xb = lists(6), lists(7)
Pa, Ka, _ = parts(Xa); Pb, Kb, _ = parts(Xb)
slope = {5: {j: WH[5, j] * abs(WI[j]) for j in HJ}, 9: {j: WH[9, j] * abs(WI[j]) for j in HJ}}
targets = {"r4": ("x",), "r7": (), "r8": (), "r1": (), "r6": (),
           "D5": ("x", "next", "g5prev"), "D9": ("x", "next", "g9prev", "g5prev")}
for kind in ("basic", "roles"):
    print(f"\n=== feature set: {kind}")
    for tg, extra in targets.items():
        A = np.concatenate([design(Xa, t, Ka, Pa[t], kind, extra) for t in range(1, 9)])
        b = np.concatenate([Pa[t][tg] for t in range(1, 9)])
        co = np.linalg.lstsq(A, b, rcond=None)[0]
        Ab = np.concatenate([design(Xb, t, Kb, Pb[t], kind, extra) for t in range(1, 9)])
        bb = np.concatenate([Pb[t][tg] for t in range(1, 9)])
        e = bb - Ab @ co
        msg = f"  {tg}: {A.shape[1]:3d} coefficients, variance matched {1 - e.var() / bb.var():.4f}, rms error {np.sqrt((e ** 2).mean()):.3f}"
        if tg[0] == "r":
            j = int(tg[1:])
            msg += f" (x hinge slope: neuron 5 {abs(slope[5][j]) * np.sqrt((e ** 2).mean()):.2f}, neuron 9 {abs(slope[9][j]) * np.sqrt((e ** 2).mean()):.2f} units)"
        print(msg)
