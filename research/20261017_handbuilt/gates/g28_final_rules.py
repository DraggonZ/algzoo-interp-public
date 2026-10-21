"""The final rules, scored on two fresh sets of 200,000 lists (seeds 9 and 10), fits on seed 6.
Keep gate (neuron 5): switch points r4, r7, r8 (and r1, r6) from m1 by its age and m2 by both ages; history part D5 from
m1/m2 by ages, x_t, x_{t+1}, its own previous value, the hinge values of each past number and both gates' past values;
pooled over steps, and with separate coefficients for each step.
Late gate (neuron 9): the short written rule
  n9(t+1) = ReLU( D9 + 1.48 (x_t - r4)^+ - 66.3 (m1 - x_t)^+ ),  r4 = 0.933 m1 + 0.012 m2,
  D9 = 1.77 m1 - 0.48 m2 - 0.08 x_t - 0.23 x_{t+1} + 0.08 n9(t)        (hinges of 1, 6, 8 left out)
and the table version (switch points and D9 by ages).
Difference to the intact network with its standard error from the lists where exactly one of the two is right."""
import pickle
import numpy as np
from gatelib import lists, run, acc_by_kind, answer
from gaterules3 import network_parts, fit_part, rule_values, HJ

co = pickle.load(open("g16_tables.pkl", "rb"))
Xf = lists(6); NP = None
D5K = "gates|-kept|-forgot|-neg|-rec|-p"
keep_pooled = {**{f"r{j}": "ages" for j in HJ}, "D5": D5K, "D9": "ages"}
keep_step = {**{f"r{j}": "ages|step" for j in HJ}, "D5": D5K + "|step", "D9": "ages"}
late_short = {"r1": "off", "r4": "basic", "r6": "off", "r7": "const:1.0", "r8": "off", "D9": "basic", "D5": "basic"}
late_table = {**{f"r{j}": "ages" for j in HJ}, "D9": "ages", "D5": "basic"}
for sp in (keep_pooled, keep_step, late_short, late_table):
    for part, kind in sp.items():
        if (part, kind) not in co and not (kind.startswith("const") or kind == "off"):
            if NP is None:
                NP = network_parts(Xf)
            co[(part, kind)] = fit_part(Xf, NP, part, kind)
pickle.dump(co, open("g28_tables.pkl", "wb"))
c = co[("D9", "basic")]; c4 = co[("r4", "basic")]
print(f"late gate short rule numbers: D9 = {c[0]:+.3f} m1 {c[1]:+.3f} m2 {c[2]:+.3f} x_t {c[3]:+.3f} x_(t+1) {c[4]:+.3f} own; r4 = {c4[0]:.3f} m1 {c4[1]:+.3f} m2")
KEYS = ["all", "S before L (revealed)", "S after L (kept)", "S at 8", "S at 9", "L at 8", "L at 9", "S right after L"]
for seed in (9, 10):
    X = lists(seed); y = answer(X); L0 = run(X)[0]; ok0 = L0.argmax(1) == y
    base = acc_by_kind(X, L0, KEYS)
    print(f"\nseed {seed}: " + " " * 26 + "  ".join(f"{k[:6]:>6s}" for k in KEYS))
    print(f"{'intact':40s}" + "  ".join(f"{100 * base[k][0]:6.2f}" for k in KEYS))
    vals = {}
    for name, sp, g in (("keep gate, pooled", keep_pooled, 5), ("keep gate, per step", keep_step, 5),
                        ("late gate, short rule", late_short, 9), ("late gate, tables", late_table, 9)):
        V = rule_values(X, sp, co)[0]; vals[name] = V[g]
        Lr = run(X, sub={g: V[g]})[0]; ok = Lr.argmax(1) == y; r = acc_by_kind(X, Lr, KEYS)
        se = np.sqrt(((ok != ok0).mean()) / len(X)) * 100
        print(f"{name:40s}" + "  ".join(f"{100 * r[k][0]:6.2f}" for k in KEYS) + f"   change {100 * (ok.mean() - ok0.mean()):+.2f} (se {se:.2f})")
    for kn in ("keep gate, pooled", "keep gate, per step"):
        Lr = run(X, sub={5: vals[kn], 9: vals["late gate, short rule"]})[0]; ok = Lr.argmax(1) == y; r = acc_by_kind(X, Lr, KEYS)
        se = np.sqrt(((ok != ok0).mean()) / len(X)) * 100
        print(f"{'both: ' + kn + ' + late short':40s}" + "  ".join(f"{100 * r[k][0]:6.2f}" for k in KEYS) + f"   change {100 * (ok.mean() - ok0.mean()):+.2f} (se {se:.2f})")
