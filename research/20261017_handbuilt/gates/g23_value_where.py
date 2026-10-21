"""Where does each gate's value have to be precise? Its push onto one group of receivers is scaled by 1.05 (5% too
large) or 0.95, at steps after records or after other numbers, with every other path unchanged. 100,000 lists (seed 8).
The readout path is scaled through the output weight of the gate (it only reads the last step)."""
import numpy as np
from gatelib import WI, WH, WO, MEM, CANC, lists, run, task_vars, acc_by_kind

X = lists(8, 100_000); V = task_vars(X); L0 = run(X)[0]
after_rec = np.zeros((len(X), 10), bool); after_rec[:, 1:] = V["new"][:, :9].astype(bool)
KEYS = ["all", "S before L (revealed)", "S after L (kept)", "S at 8", "L at 8", "S right after L"]
base = acc_by_kind(X, L0, KEYS)
print(" " * 58 + "  ".join(f"{k[:6]:>6s}" for k in KEYS))
print(f"{'intact':58s}" + "  ".join(f"{100 * base[k][0]:6.1f}" for k in KEYS))


def scaled_run(g, recv, s, when=None, readout=False):
    def hook(t, hp, pre, h):
        if t == 0 or not recv:
            return h
        m = np.ones(len(X), bool) if when is None else when[:, t - 1]
        corr = np.outer(hp[:, g] * (s - 1) * m, WH[recv, g])
        h[:, recv] = np.maximum(pre[:, recv] + corr, 0); return h
    wo = WO.copy()
    if readout:
        wo[:, g] *= s
    return run(X, params=(WI, WH, wo), hook=hook)[0]


for g in (5, 9):
    for s in (1.05, 0.95):
        for name, recv, ro in (("readout only", [], True), ("memory", MEM, False), ("neuron 0", [0], False),
                               ("memory except 0", [m for m in MEM if m != 0], False), ("hiding neurons", CANC, False),
                               ("itself", [g], False), ("the other gate", [14 - g], False)):
            r = acc_by_kind(X, scaled_run(g, recv, s, readout=ro), KEYS)
            print(f"{f'neuron {g} push x {s} onto {name}':58s}" + "  ".join(f"{100 * r[k][0]:6.1f}" for k in KEYS), flush=True)
    for name, when in (("after records", after_rec), ("after other numbers", ~after_rec)):
        r = acc_by_kind(X, scaled_run(g, list(range(16)), 1.05, when=when), KEYS)
        print(f"{f'neuron {g} push x 1.05 onto all neurons, {name}':58s}" + "  ".join(f"{100 * r[k][0]:6.1f}" for k in KEYS), flush=True)
