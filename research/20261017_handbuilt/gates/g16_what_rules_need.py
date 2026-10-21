"""Which history items does each gate rule need? Substitution (one gate by its rule, the rest on its own weights),
100,000 fresh lists (seed 8); fits on 200,000 lists (seed 6). Switch points by 'ages' (m1 by its age, m2 by both ages)
unless stated. Neuron 5: the history part D5 at full detail ('gates') with one group of history items left out at a
time. Neuron 9: switch points and D9 at the least detail that still passes."""
import pickle
import numpy as np
from gatelib import lists, run, acc_by_kind
from gaterules3 import network_parts, fit_part, rule_values, HJ

Xf = lists(6); NP = network_parts(Xf)
co = pickle.load(open("g11_tables.pkl", "rb"))
X = lists(8, 100_000); L0 = run(X)[0]
KEYS = ["all", "S before L (revealed)", "S after L (kept)", "S at 8", "S at 9", "L at 8", "L at 9", "S right after L"]
base = acc_by_kind(X, L0, KEYS)
print(" " * 52 + "  ".join(f"{k[:6]:>6s}" for k in KEYS))
print(f"{'intact':52s}" + "  ".join(f"{100 * base[k][0]:6.1f}" for k in KEYS))


def spec(r="ages", D5="gates", D9="ages", **over):
    s = {f"r{j}": r for j in HJ}; s["D5"] = D5; s["D9"] = D9; s.update(over); return s


def need(sp):
    for part, kind in sp.items():
        if (part, kind) not in co and not (kind.startswith("const") or kind == "off"):
            co[(part, kind)] = fit_part(Xf, NP, part, kind)


D5_ALL = "gates|-kept|-forgot|-neg|-rec|-p"
variants = [("5", "D5 full", spec()),
            ("5", "D5 without other kept numbers", spec(D5="gates|-kept")),
            ("5", "D5 without forgotten numbers", spec(D5="gates|-forgot")),
            ("5", "D5 without negative numbers", spec(D5="gates|-neg")),
            ("5", "D5 without older beaten records", spec(D5="gates|-rec")),
            ("5", "D5 without p (when p is not m2)", spec(D5="gates|-p")),
            ("5", "D5 without past hinge values", spec(D5="gates|-hz")),
            ("5", "D5 without past gate values", spec(D5="gates|-gz")),
            ("5", "D5 without own previous value", spec(D5="gates|-own")),
            ("5", "D5: m1, m2 by ages + hinges + gates only", spec(D5=D5_ALL)),
            ("5", "D5: roles + past gate values (no hinges)", spec(D5="gates|-hz")),
            ("9", "switch points ages, D9 ages", spec()),
            ("9", "D9 basic", spec(D9="basic")),
            ("9", "r7 by ages, other switch points basic", spec(r="basic", r7="ages")),
            ("9", "r7, r8 by ages, others basic", spec(r="basic", r7="ages", r8="ages")),
            ("9", "r7, r4 by ages, others basic", spec(r="basic", r7="ages", r4="ages")),
            ("9", "r7, r8, r4 by ages, r1, r6 basic", spec(r="ages", r1="basic", r6="basic")),
            ("9", "hinges of 1 and 6 left out", spec(r1="off", r6="off")),
            ("9", "hinge of 8 left out", spec(r8="off")),
            ("9", "hinges of 1, 6, 8 left out", spec(r1="off", r6="off", r8="off")),
            ("9", "r7 = m1, others ages, D9 ages", spec(r7="const:1.0")),
            ("9", "D9 basic, r7, r8, r4 ages, r1, r6 basic", spec(r1="basic", r6="basic", D9="basic")),
            ]
for who, name, sp in variants:
    need(sp)
    V, R = rule_values(X, sp, co)
    sub = {5: V[5]} if who == "5" else {9: V[9]}
    Lr = run(X, sub=sub)[0]; r = acc_by_kind(X, Lr, KEYS)
    print(f"{('n' + who + ': ' + name)[:52]:52s}" + "  ".join(f"{100 * r[k][0]:6.1f}" for k in KEYS) +
          f"   agree {100 * (Lr.argmax(1) == L0.argmax(1)).mean():.1f}%", flush=True)
pickle.dump(co, open("g16_tables.pkl", "wb"))
