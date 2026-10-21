"""Neuron 5's and neuron 9's pre-activation one step after x_t, split into the pushes from each source neuron
(weight x that neuron's value after x_t), by kind of step. 200,000 lists, seed 3; steps t = 1..8 (so x_{t+1} exists).
Kinds of step: x_t a record; x_t not a record and the gate on; x_t not a record, positive, gate off."""
import numpy as np
from gatelib import WI, WH, MEM, lists, run, task_vars

X = lists(3); L, P, H = run(X); V = task_vars(X)
T = range(1, 9)
SRC = {"det 4": [4], "canc 7": [7], "canc 8": [8], "canc 1,6": [1, 6], "store 2": [2], "self": [5], "other gate": [9],
       "memory": MEM}
for g in (5, 9):
    print(f"\n===== neuron {g}: pushes onto its pre-activation one step after x_t (mean, and sd in brackets)")
    rows = {}
    for t in T:
        h = H[:, t]; pre = P[:, t + 1, g]
        new = V["new"][:, t].astype(bool); on = H[:, t + 1, g] > 0; xp = X[:, t] > 0
        masks = {"record": new, "not record, on": ~new & on, "not record, x>0, off": ~new & ~on & xp,
                 "x<0": X[:, t] < 0}
        for k, m in masks.items():
            d = {s: (h[:, idx] * WH[g, idx]).sum(1)[m] for s, idx in SRC.items()}
            d["next input"] = WI[g] * X[m, t + 1]; d["total"] = pre[m]
            for s in d:
                rows.setdefault(k, {}).setdefault(s, []).append(d[s])
    for k, d in rows.items():
        print(f"  {k} ({sum(len(a) for a in d['total'])} steps)")
        print("    " + "  ".join(f"{s}: {np.concatenate(a).mean():6.2f} ({np.concatenate(a).std():5.2f})" for s, a in d.items()))
# a few representative lists
np.set_printoptions(precision=2, suppress=True, linewidth=220)
print("\nexamples: list, then per step x_t, m2 before x_t, neuron 5 one step later, pushes det/7/8/self/9/memory/next")
for i in range(4):
    print(" list", np.round(X[i], 2))
    for t in T:
        h = H[i, t]
        parts = [WH[5, 4] * h[4], WH[5, 7] * h[7], WH[5, 8] * h[8], WH[5, 5] * h[5], WH[5, 9] * h[9], WH[5, MEM] @ h[MEM], WI[5] * X[i, t + 1]]
        print(f"  t={t} x={X[i, t]:5.2f} m1={V['m1'][i, t]:4.2f} m2={V['m2'][i, t]:4.2f} n5={H[i, t + 1, 5]:5.2f} pre={P[i, t + 1, 5]:6.2f} | " + " ".join(f"{p:6.2f}" for p in parts))
