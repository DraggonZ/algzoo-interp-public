"""Substitution test of the gate rules (gaterules.py): neuron 5 or 9 replaced at every step by its rule, the other 15
neurons on their own weights. Tables fitted on 200,000 lists (seed 6); scored on 100,000 fresh lists (seed 8).
Levels: true switch points and true D (identity check); table switch points with true D; true switch points with table D;
both from tables (input-only rule: the list and the rule's own past values)."""
import pickle
import numpy as np
from gatelib import lists, run, acc_by_kind
from gaterules import fit_tables, substitute

co = fit_tables(lists(6))
pickle.dump(co, open("g08_tables.pkl", "wb"))
X = lists(8, 100_000); L0 = run(X)[0]
KEYS = ["all", "S before L (revealed)", "S after L (kept)", "S at 8", "S at 9", "L at 8", "L at 9", "S right after L"]
base = acc_by_kind(X, L0, KEYS)
print(" " * 40 + "  ".join(f"{k[:6]:>6s}" for k in KEYS))
print(f"{'intact':40s}" + "  ".join(f"{100 * base[k][0]:6.1f}" for k in KEYS))
for g in (5, 9):
    for refs, D in (("true", "true"), ("table", "true"), ("true", "table"), ("table", "table")):
        Lr, Hr = substitute(X, g, co, refs=refs, D=D)
        r = acc_by_kind(X, Lr, KEYS)
        agree = (Lr.argmax(1) == L0.argmax(1)).mean()
        print(f"{f'neuron {g}: switch points {refs}, D {D}':40s}" + "  ".join(f"{100 * r[k][0]:6.1f}" for k in KEYS) + f"   agree {100 * agree:.1f}%")
