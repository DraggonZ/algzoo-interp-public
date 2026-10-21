"""A short written rule for the late gate (neuron 9), and what its value carries.
Rule: n9(t+1) = ReLU( D9 + 1.48 (x_t - r4)^+ - 66.3 (r7 - x_t)^+ [- 13.4 (r8 - x_t)^+ - 3.8 (r1 - x_t)^+ - 1.0 (r6 - x_t)^+] )
Variants from detailed to short; substitution on 100,000 fresh lists (seed 8), fits on seed 6 (g16_tables.pkl).
Then: the late gate's value one step after a record, fitted on the new largest m, the value it overtook p, the old
second-largest and the next input (200,000 random lists, seed 19), and the share of steps on."""
import pickle
import numpy as np
from gatelib import lists, run, acc_by_kind, task_vars
from gaterules3 import rule_values, HJ, network_parts, fit_part

co = pickle.load(open("g16_tables.pkl", "rb"))
Xf = lists(6); NP = None
X = lists(8, 100_000); L0 = run(X)[0]
KEYS = ["all", "S before L (revealed)", "S after L (kept)", "S at 8", "L at 8", "L at 9", "S right after L"]
base = acc_by_kind(X, L0, KEYS)
print(" " * 56 + "  ".join(f"{k[:6]:>6s}" for k in KEYS))
print(f"{'intact':56s}" + "  ".join(f"{100 * base[k][0]:6.1f}" for k in KEYS))
variants = [
    ("switch points m1, m2 by ages; D9 by ages (passes)", dict(r1="ages", r4="ages", r6="ages", r7="ages", r8="ages", D9="ages")),
    ("r7 = m1; r4 by ages; 1, 6, 8 left out; D9 by ages", dict(r1="off", r4="ages", r6="off", r7="const:1.0", r8="off", D9="ages")),
    ("r7 = m1; r4 basic; 1, 6, 8 left out; D9 basic", dict(r1="off", r4="basic", r6="off", r7="const:1.0", r8="off", D9="basic")),
    ("r7 = m1; r4 = 0.93 m1; r8 by ages; 1, 6 left out; D9 basic", dict(r1="off", r4="basic", r6="off", r7="const:1.0", r8="ages", D9="basic")),
    ("r7 by ages; r4 basic; 1, 6, 8 left out; D9 basic", dict(r1="off", r4="basic", r6="off", r7="ages", r8="off", D9="basic")),
    ("r7 by ages; r4 by ages; 1, 6, 8 left out; D9 basic", dict(r1="off", r4="ages", r6="off", r7="ages", r8="off", D9="basic")),
]
for name, sp in variants:
    sp = {**sp, "D5": "basic"}
    for part, kind in sp.items():
        if (part, kind) not in co and not (kind.startswith("const") or kind == "off"):
            if NP is None:
                NP = network_parts(Xf)
            co[(part, kind)] = fit_part(Xf, NP, part, kind)
    V, R = rule_values(X, sp, co)
    Lr = run(X, sub={9: V[9]})[0]; r = acc_by_kind(X, Lr, KEYS)
    print(f"{name[:56]:56s}" + "  ".join(f"{100 * r[k][0]:6.1f}" for k in KEYS) + f"   agree {100 * (Lr.argmax(1) == L0.argmax(1)).mean():.1f}%", flush=True)
c = co[("D9", "basic")]; c4 = co[("r4", "basic")]
print(f"basic parts: D9 = {c[0]:+.2f} m1 {c[1]:+.2f} m2 {c[2]:+.2f} x_t {c[3]:+.2f} x_(t+1) {c[4]:+.2f} own previous; r4 = {c4[0]:.3f} m1 {c4[1]:+.3f} m2")

Z = lists(19); V = task_vars(Z); _, P, H = run(Z)
rows = []
for t in range(1, 9):
    m = V["new"][:, t].astype(bool) & (V["m1"][:, t] > 0)
    rows.append(np.c_[Z[m, t], V["m1"][m, t], V["m2"][m, t], Z[m, t + 1], H[m, t + 1, 9]])
R = np.concatenate(rows); A = R[:, :4]; b = R[:, 4]
co9 = np.linalg.lstsq(A, b, rcond=None)[0]; e = b - A @ co9
print(f"\nlate gate one step after a record (with a positive earlier largest p), {len(b)} steps:")
print(f"  value = {co9[0]:.2f} m {co9[1]:+.2f} p {co9[2]:+.2f} (old second) {co9[3]:+.2f} x_next   (variance matched {1 - e.var() / b.var():.3f}; mean value {b.mean():.2f})")
for lo, hi in ((0, 0.1), (0.1, 0.3), (0.3, 1), (1, 9)):
    s = (A[:, 0] - A[:, 1] >= lo) & (A[:, 0] - A[:, 1] < hi)
    print(f"  jump m - p in [{lo}, {hi}): on {np.mean(b[s] > 0):.3f}, mean value {b[s].mean():.2f}, mean m {A[s, 0].mean():.2f}")
