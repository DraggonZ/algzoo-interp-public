"""Does the keep gate's rule need its detail for the decision (keep or forget) or for the size of what it writes?
Decision-only tests: neuron 5 takes its own pre-activation from the weights (pre x mask, linear while on) but its on/off
mask from a rule. Value-only tests: on/off from the network itself, value from a rule (scaled, or simplified).
100,000 fresh lists (seed 8); rules fitted on seed 6 (g16_tables.pkl)."""
import pickle
import numpy as np
from gatelib import lists, run, acc_by_kind, task_vars
from gaterules3 import rule_values, HJ

co = pickle.load(open("g16_tables.pkl", "rb"))
X = lists(8, 100_000); L0, P0, H0 = run(X); V = task_vars(X)
KEYS = ["all", "S before L (revealed)", "S after L (kept)", "S at 8", "L at 8", "S right after L"]
base = acc_by_kind(X, L0, KEYS)
print(" " * 56 + "  ".join(f"{k[:6]:>6s}" for k in KEYS))
print(f"{'intact':56s}" + "  ".join(f"{100 * base[k][0]:6.1f}" for k in KEYS))


def show(name, L):
    r = acc_by_kind(X, L, KEYS)
    print(f"{name[:56]:56s}" + "  ".join(f"{100 * r[k][0]:6.1f}" for k in KEYS) + f"   agree {100 * (L.argmax(1) == L0.argmax(1)).mean():.1f}%", flush=True)


full = {**{f"r{j}": "ages" for j in HJ}, "D5": "gates|-kept|-forgot|-neg|-rec|-p", "D9": "ages"}
basic = {**{f"r{j}": "basic" for j in HJ}, "D5": "basic", "D9": "basic"}
Vf = rule_values(X, full, co)[0][5]; Vb = rule_values(X, basic, co)[0][5]
# masks one step after each number (steps 1..9); step 0 from the network
rec = np.zeros((len(X), 10), bool); rec[:, 1:] = V["new"][:, :9].astype(bool)
m2 = np.zeros((len(X), 10)); m2[:, 1:] = V["m2"][:, :9]
xs = np.zeros((len(X), 10)); xs[:, 1:] = X[:, :9]
masks = {"network's own": H0[:, :, 5] > 0,
         "full rule's": Vf > 0, "basic rule's": Vb > 0}
for c in (0.65, 0.7, 0.75, 0.8):
    mk = rec | (xs >= c * m2) & (xs > 0); mk[:, 0] = H0[:, 0, 5] > 0
    masks[f"record or x >= {c} x second-largest"] = mk
for name, mk in masks.items():
    def hook(t, hp, pre, h, mk=mk):
        h[:, 5] = pre[:, 5] * mk[:, t]; return h
    show(f"decision {name}, value from weights", run(X, hook=hook)[0])
for name, mk in masks.items():
    print(f"  mask {name}: differs from the network's on/off on {100 * (mk != (H0[:, :, 5] > 0)).mean():.2f}% of steps")
# value-only
on = H0[:, :, 5] > 0
for name, val in (("full rule", Vf), ("basic rule", Vb)):
    def hook(t, hp, pre, h, val=val):
        h[:, 5] = np.where(on[:, t], np.maximum(val[:, t], 0.0), 0.0); return h
    show(f"on/off from network, value from {name} (no feedback)", run(X, hook=hook)[0])
for s in (0.9, 0.95, 1.05, 1.1):
    def hook(t, hp, pre, h, s=s):
        h[:, 5] = np.maximum(pre[:, 5], 0) * s; return h
    show(f"own value scaled by {s}", run(X, hook=hook)[0])
