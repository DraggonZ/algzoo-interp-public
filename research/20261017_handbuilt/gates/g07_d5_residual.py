"""Where does the role table miss neuron 5's history part D5? Fit as in g06 (roles + x_t, x_{t+1}, own previous value),
then split the test-set error by step, by age of the largest, by which memory neurons are off at step t, and by
whether the late gate was on at step t. Also: the same fit on D5 with the late gate's push (0.30 x neuron 9) removed
and with each memory neuron's push removed in turn, to see which part carries the error."""
import numpy as np
from gatelib import WI, WH, MEM, lists, run
from rolefeat import role_features

HJ = [1, 4, 6, 7, 8]


def build(X):
    L, P, H = run(X)
    keep = np.zeros((len(X), 10), bool); keep[:, :9] = H[:, 1:, 5] > 0
    rows = []
    for t in range(1, 9):
        F, names, info = role_features(X, t, keep)
        A = np.c_[F, X[:, t], X[:, t + 1], H[:, t, 5]]
        parts = {"D5": P[:, t + 1, 5] - sum(WH[5, j] * H[:, t, j] for j in HJ),
                 "late gate push": WH[5, 9] * H[:, t, 9], "self push": WH[5, 5] * H[:, t, 5],
                 "store push": WH[5, 2] * H[:, t, 2]}
        for j in MEM:
            parts[f"mem {j}"] = WH[5, j] * H[:, t, j]
        rows.append((t, A, parts, info, H[:, t]))
    return rows


ra, rb = build(lists(6)), build(lists(7))
A = np.concatenate([r[1] for r in ra]); Ab = np.concatenate([r[1] for r in rb])
tb = np.concatenate([np.full(len(r[1]), r[0]) for r in rb])
a1b = np.concatenate([r[3]["a1"] for r in rb])
Hb = np.concatenate([r[4] for r in rb])
for tg in ["D5", "late gate push", "store push"] + [f"mem {j}" for j in MEM]:
    b = np.concatenate([r[2][tg] for r in ra]); bb = np.concatenate([r[2][tg] for r in rb])
    co = np.linalg.lstsq(A, b, rcond=None)[0]; e = bb - Ab @ co
    print(f"{tg:15s} sd {bb.std():6.3f}  rms error of role fit {np.sqrt((e ** 2).mean()):.3f}")
    if tg == "D5":
        print("  coefficient on own previous value (weight is -0.42):", round(co[-1], 3))
        print("  rms error by step t: " + " ".join(f"{t}:{np.sqrt((e[tb == t] ** 2).mean()):.2f}" for t in range(1, 9)))
        print("  rms error by age of largest a1: " + " ".join(f"{a}:{np.sqrt((e[a1b == a] ** 2).mean()):.2f}" for a in range(0, 9)))
        print("  rms error by late gate on at step t: off %.2f, on %.2f" % (np.sqrt((e[Hb[:, 9] == 0] ** 2).mean()), np.sqrt((e[Hb[:, 9] > 0] ** 2).mean())))
        for j in MEM:
            off = Hb[:, j] == 0
            print(f"  memory neuron {j:2d} off at step t on {off.mean():.2f} of steps: rms error off {np.sqrt((e[off] ** 2).mean()):.2f}, on {np.sqrt((e[~off] ** 2).mean()):.2f}")
        eD = e
