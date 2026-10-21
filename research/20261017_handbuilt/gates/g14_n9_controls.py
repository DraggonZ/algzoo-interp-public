"""Controls for the late gate's job (neuron 9). 200,000 random lists (seed 14).
(1) Its push onto the hiding neurons (weights -0.69 onto 1, -0.64 onto 6, -0.07 onto 8): removed, scaled, sign-flipped,
    moved onto other hiding neurons (7 and 8, or 1 and 6 swapped), its value shuffled across lists on this path only.
(2) Its readout: on lists with the largest at position 8 (neuron 9 fires at the last step), what the network names when
    the readout is removed, and which readout weights matter.
(3) Where the hiding push acts: the change in the largest's final score when the path is removed, split by the readout
    of each neuron group (hiding neurons, memory, other)."""
import numpy as np
from gatelib import WI, WH, WO, MEM, CANC, lists, run, answer, acc_by_kind

X = lists(14); y = answer(X); L0, P0, H0 = run(X)
KEYS = ["all", "S before L (revealed)", "S after L (kept)", "L at 8", "L at 9", "S right after L"]
base = acc_by_kind(X, L0, KEYS)
print(" " * 46 + "  ".join(f"{k[:6]:>6s}" for k in KEYS))
print(f"{'intact':46s}" + "  ".join(f"{100 * base[k][0]:6.1f}" for k in KEYS))


def show(name, L):
    r = acc_by_kind(X, L, KEYS)
    print(f"{name[:46]:46s}" + "  ".join(f"{100 * r[k][0]:6.1f}" for k in KEYS))


w9 = WH[CANC, 9].copy()   # onto 1, 6, 7, 8
for name, col in (("removed", np.zeros(4)), ("scaled 0.5", 0.5 * w9), ("scaled 1.5", 1.5 * w9), ("sign flipped", -w9),
                  ("moved onto 7 and 8 (1->7, 6->8)", np.array([0, 0, w9[0], w9[1]])),
                  ("1 and 6 swapped", np.array([w9[1], w9[0], 0, w9[3]])),
                  ("onto 1 only", np.array([w9[0], 0, 0, 0])), ("onto 6 only", np.array([0, w9[1], 0, 0]))):
    wh = WH.copy(); wh[CANC, 9] = col
    show(f"push onto hiding neurons {name}", run(X, params=(WI, wh, WO))[0])
# shuffled value on this path only: the hiding neurons receive neuron 9's value from another list
rng = np.random.default_rng(0); perm = rng.permutation(len(X))
sh = H0[perm, :, 9]


def hook(t, hp, pre, h):
    if t == 0:
        return h
    corr = np.outer(sh[:, t - 1] - hp[:, 9], WH[CANC, 9])
    h[:, CANC] = np.maximum(pre[:, CANC] + corr, 0); return h


show("hiding neurons get neuron 9 of another list", run(X, hook=hook)[0])
# also the same-size push given at a wrong time: one step later
def hook_late(t, hp, pre, h, store={}):
    if t == 0:
        store["prev9"] = np.zeros(len(X)); return h
    corr = np.outer(store.get("prev9", np.zeros(len(X))) - hp[:, 9], WH[CANC, 9])
    store["prev9"] = hp[:, 9].copy()
    h[:, CANC] = np.maximum(pre[:, CANC] + corr, 0); return h


show("hiding neurons get neuron 9 one step late", run(X, hook=hook_late)[0])

print("\n(2) readout, lists with the largest at position 8")
o = X.argsort(1); iL, iS = o[:, -1], o[:, -2]
m8 = iL == 8
wo = WO.copy(); wo[:, 9] = 0; L1 = run(X, params=(WI, WH, wo))[0]
bad = m8 & (L0.argmax(1) == y) & (L1.argmax(1) != y)
print(f"  largest at 8: intact {100 * (L0.argmax(1) == y)[m8].mean():.1f}%, readout removed {100 * (L1.argmax(1) == y)[m8].mean():.1f}%; "
      f"broken lists name position 8 in {100 * (L1.argmax(1)[bad] == 8).mean():.1f}%")
for k in range(10):
    wo = WO.copy(); wo[k, 9] = 0; Lk = run(X, params=(WI, WH, wo))[0]
    print(f"  remove only the readout weight onto position {k} ({WO[k, 9]:+.2f}): largest at 8 {100 * (Lk.argmax(1) == y)[m8].mean():.1f}%, all {100 * (Lk.argmax(1) == y).mean():.2f}%")
print(f"  neuron 9 at the last step: on {(H0[:, 9, 9] > 0)[m8].mean():.3f} of lists with the largest at 8, {(H0[:, 9, 9] > 0)[~m8].mean():.3f} of others")

print("\n(3) change in the largest's final score when the hiding push is removed, by readout group (revealed lists, largest at 1-7)")
wh = WH.copy(); wh[CANC, 9] = 0; _, _, H1 = run(X, params=(WI, wh, WO))
sel = (iS < iL) & (iL <= 7)
d = H1[sel, 9] - H0[sel, 9]                 # final state change (B, 16)
li = iL[sel]
for name, grp in (("hiding 1,6,7,8", CANC), ("memory", MEM), ("store+detector", [2, 4]), ("gates 5,9", [5, 9])):
    ch = (d[:, grp] * WO[li][:, grp]).sum(1)
    print(f"  {name:15s}: mean change of the largest's score {ch.mean():+6.2f}")
chS = (d * WO[iS[sel]]).sum(1); chL = (d * WO[li]).sum(1)
print(f"  total: largest's score {chL.mean():+.2f}, second's score {chS.mean():+.2f}")
