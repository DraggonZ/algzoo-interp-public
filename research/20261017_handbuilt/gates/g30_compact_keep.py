"""Shorter versions of the keep-gate rule, and both gates replaced together with the table rules.
Seed 9 (200,000 fresh lists); fits on seed 6. 'm1ages' = m1 by its age only (9 numbers per switch point).
Neuron 5 barely reads hiding neurons 1 and 6 directly (hinge slopes -0.01 and +0.17), but their past values enter
its history part D5 (the memory's echo), so their switch points are still needed; here they are made coarser instead.
(An earlier version of this script left them out, which also zeroed those history features: 45%, not a valid test.)"""
import pickle
import numpy as np
from gatelib import lists, run, acc_by_kind, answer
from gaterules3 import network_parts, fit_part, rule_values, HJ

co = pickle.load(open("g28_tables.pkl", "rb"))
Xf = lists(6); NP = None
D5K = "gates|-kept|-forgot|-neg|-rec|-p"
full = {**{f"r{j}": "ages" for j in HJ}, "D5": D5K, "D9": "ages"}
variants = [
    ("keep: full (switch points by ages)", full, (5,)),
    ("keep: r1, r6 basic", {**full, "r1": "basic", "r6": "basic"}, (5,)),
    ("keep: r4, r7 from m1 by age only", {**full, "r4": "m1ages", "r7": "m1ages"}, (5,)),
    ("keep: r4, r7 from m1 by age; r1, r6 basic", {**full, "r1": "basic", "r6": "basic", "r4": "m1ages", "r7": "m1ages"}, (5,)),
    ("keep: r8 from m1 by age only", {**full, "r8": "m1ages"}, (5,)),
    ("keep: all switch points from m1 by age only", {**full, **{f"r{j}": "m1ages" for j in HJ}}, (5,)),
    ("both: keep full + late tables", full, (5, 9)),
]
X = lists(9); y = answer(X); L0 = run(X)[0]; ok0 = L0.argmax(1) == y
KEYS = ["all", "S before L (revealed)", "S after L (kept)", "S at 8", "L at 8", "S right after L"]
base = acc_by_kind(X, L0, KEYS)
print(" " * 54 + "  ".join(f"{k[:6]:>6s}" for k in KEYS))
print(f"{'intact':54s}" + "  ".join(f"{100 * base[k][0]:6.2f}" for k in KEYS))
for name, sp, gs in variants:
    for part, kind in sp.items():
        if (part, kind) not in co and not (kind.startswith("const") or kind == "off"):
            if NP is None:
                NP = network_parts(Xf)
            co[(part, kind)] = fit_part(Xf, NP, part, kind)
    V = rule_values(X, sp, co)[0]
    Lr = run(X, sub={g: V[g] for g in gs})[0]; ok = Lr.argmax(1) == y; r = acc_by_kind(X, Lr, KEYS)
    se = np.sqrt(((ok != ok0).mean()) / len(X)) * 100
    print(f"{name[:54]:54s}" + "  ".join(f"{100 * r[k][0]:6.2f}" for k in KEYS) + f"   change {100 * (ok.mean() - ok0.mean()):+.2f} (se {se:.2f})", flush=True)
