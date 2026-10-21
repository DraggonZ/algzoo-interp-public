"""What the keep gate (neuron 5) does at records and at other numbers, path by path.
Neuron 5 fires one step after every record and after non-record numbers above about 0.75 x the second-largest.
Here its outgoing pushes are removed only at the steps after records, or only at the steps after non-records,
for all receivers or one group of receivers (memory, neuron 0 alone, hiding neurons, itself, readout).
200,000 random lists (seed 13), scored by kind of list. The readout only reads the last step (x_8's gate)."""
import numpy as np
from gatelib import WI, WH, WO, MEM, CANC, lists, run, task_vars, acc_by_kind

X = lists(13); V = task_vars(X); L0, P0, H0 = run(X)
rec_next = np.zeros((len(X), 10), bool); rec_next[:, 1:] = V["new"][:, :9].astype(bool)   # step t+1 after a record x_t
KEYS = ["all", "S before L (revealed)", "S after L (kept)", "S at 8", "L at 8", "L at 9", "S right after L", "S after L, S at 9"]
base = acc_by_kind(X, L0, KEYS)
print(" " * 50 + "  ".join(f"{k[:6]:>6s}" for k in KEYS))
print(f"{'intact':50s}" + "  ".join(f"{100 * base[k][0]:6.1f}" for k in KEYS))
RECV = {"all receivers": list(range(16)), "memory": MEM, "neuron 0": [0], "memory except 0": [m for m in MEM if m != 0],
        "hiding neurons": CANC, "itself": [5]}


def removed_run(when, recv, readout=False):
    """Remove neuron 5's push onto `recv` at the steps where `when` (B, 10) is true: the receivers' pre-activations at
    the next step are computed as if neuron 5 were 0 there."""
    def hook(t, hp, pre, h):
        if t == 0:
            return h
        m = when[:, t - 1]
        if m.any():
            corr = -np.outer(hp[m, 5], WH[recv, 5])          # undo neuron 5's push from the previous step
            newpre = pre[m][:, recv] + corr
            hh = h[m]; hh[:, recv] = np.maximum(newpre, 0); h[m] = hh
        return h
    # this hook changes the receivers at step t using neuron 5 at step t-1; pre includes the push, so subtract it
    wo = WO.copy()
    if readout:
        wo[:, 5] = 0
    L = run(X, params=(WI, WH, wo), hook=hook)[0]
    return L


for wname, when in (("after records", rec_next), ("after non-records", ~rec_next)):
    for rname, recv in RECV.items():
        L = removed_run(when, recv)
        r = acc_by_kind(X, L, KEYS)
        print(f"{('remove at steps ' + wname + ', onto ' + rname)[:50]:50s}" + "  ".join(f"{100 * r[k][0]:6.1f}" for k in KEYS))
# note: the hook removes the push one step at a time on the running state, so downstream effects propagate.
