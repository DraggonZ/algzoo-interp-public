"""What each gate is for: remove it, or one of its output paths, and score by kind of list (200,000 lists, seed 2).
Removing a path = setting that neuron's outgoing weights to zero for the chosen receivers (readout or recurrent).
Controls: the gate's own values shuffled across lists (same step, same distribution, wrong list), and the per-step mean."""
import numpy as np
from gatelib import WI, WH, WO, MEM, CANC, lists, run, acc_by_kind

X = lists(2); L0, P0, H0 = run(X)
base = acc_by_kind(X, L0)
KEYS = ["all", "S before L (revealed)", "S after L (kept)", "S at 8", "S at 9", "L at 8", "L at 9", "S right after L",
        "S after L, S at 8", "S before L, L at 9"]
rng = np.random.default_rng(0)


def show(name, L):
    r = acc_by_kind(X, L, KEYS)
    print(f"{name:44s} " + "  ".join(f"{100 * r[k][0]:5.1f}" for k in KEYS))


print(" " * 45 + "  ".join(f"{k[:5]:>5s}" for k in KEYS))
print("kinds: " + "; ".join(f"{k} ({base[k][1]} lists)" for k in KEYS))
show("intact", L0)
for g in (5, 9):
    print(f"--- neuron {g}")
    show(f"{g} zero at every step", run(X, sub={g: np.zeros((len(X), 10))})[0])
    show(f"{g} per-step mean", run(X, sub={g: np.broadcast_to(H0[:, :, g].mean(0), (len(X), 10))})[0])
    sh = H0[rng.permutation(len(X)), :, g]
    show(f"{g} values shuffled across lists", run(X, sub={g: sh})[0])
    wo = WO.copy(); wo[:, g] = 0
    show(f"{g} readout removed", run(X, params=(WI, WH, wo))[0])
    for name, rec in (("all recurrent outputs", [n for n in range(16)]), ("to memory", MEM), ("to cancellers", CANC),
                      ("to itself", [g]), ("to the other gate", [14 - g]), ("to store+detector", [2, 4]),
                      ("to neuron 0 only", [0])):
        wh = WH.copy(); wh[rec, g] = 0
        show(f"{g} {name} removed", run(X, params=(WI, wh, WO))[0])
    # timing: zero the gate only at chosen steps
    for steps in ([9], [8, 9], list(range(1, 9))):
        z = H0[:, :, g].copy(); z[:, steps] = 0
        show(f"{g} zero at steps {steps}", run(X, sub={g: z})[0])
# readout share of each position from the gates at the last step
print("\nreadout weights of neuron 5 by position:", np.round(WO[:, 5], 2))
print("readout weights of neuron 9 by position:", np.round(WO[:, 9], 2))
print(f"neuron 5 at the last step: mean {H0[:, 9, 5].mean():.2f}, on {(H0[:, 9, 5] > 0).mean():.3f}")
print(f"neuron 9 at the last step: mean {H0[:, 9, 9].mean():.2f}, on {(H0[:, 9, 9] > 0).mean():.3f}")
