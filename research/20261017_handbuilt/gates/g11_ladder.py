"""Simplification ladder: which parts of each gate rule need detail? Substitute one gate (or both) by a rule whose
switch points and history part use simpler feature sets (gaterules3.py), and score on 100,000 fresh lists (seed 8).
Fits on 200,000 lists (seed 6). Also prints the rms error of each fitted part on the fitting lists."""
import pickle
import time
import numpy as np
from gatelib import lists, run, acc_by_kind
from gaterules3 import network_parts, fit_part, rule_values, HJ

Xf = lists(6); NP = network_parts(Xf)
co = {}
t0 = time.time()
for j in HJ:
    for kind in ("basic", "ages", "roles"):
        co[(f"r{j}", kind)] = fit_part(Xf, NP, f"r{j}", kind)
for g in (5, 9):
    for kind in ("basic", "ages", "roles", "hinges", "gates"):
        co[(f"D{g}", kind)] = fit_part(Xf, NP, f"D{g}", kind)
pickle.dump(co, open("g11_tables.pkl", "wb"))
print(f"fitted in {time.time() - t0:.0f} s; basic coefficients:")
for j in HJ:
    print(f"  r{j} = {co[(f'r{j}', 'basic')][0]:+.3f} m1 {co[(f'r{j}', 'basic')][1]:+.3f} m2")
for g in (5, 9):
    c = co[(f"D{g}", "basic")]
    print(f"  D{g} = {c[0]:+.2f} m1 {c[1]:+.2f} m2 {c[2]:+.2f} x_t {c[3]:+.2f} x_(t+1) {c[4]:+.2f} own previous value")

X = lists(8, 100_000); L0 = run(X)[0]
KEYS = ["all", "S before L (revealed)", "S after L (kept)", "S at 8", "S at 9", "L at 8", "L at 9", "S right after L"]
base = acc_by_kind(X, L0, KEYS)
print("\n" + " " * 52 + "  ".join(f"{k[:6]:>6s}" for k in KEYS))
print(f"{'intact':52s}" + "  ".join(f"{100 * base[k][0]:6.1f}" for k in KEYS))


def spec(r="roles", D5="gates", D9="gates", **over):
    s = {f"r{j}": r for j in HJ}; s["D5"] = D5; s["D9"] = D9; s.update(over); return s


variants = [
    ("5", "full (switch points roles, D gates)", spec()),
    ("5", "switch points by ages", spec(r="ages")),
    ("5", "switch points basic", spec(r="basic")),
    ("5", "r4 basic", spec(r4="basic")), ("5", "r7 basic", spec(r7="basic")), ("5", "r8 basic", spec(r8="basic")),
    ("5", "r7 = m1", spec(r7="const:1.0")),
    ("5", "D5 hinges", spec(D5="hinges")), ("5", "D5 roles", spec(D5="roles")), ("5", "D5 ages", spec(D5="ages")),
    ("5", "D5 basic", spec(D5="basic")),
    ("5", "all basic", spec(r="basic", D5="basic", D9="basic")),
    ("9", "full", spec()),
    ("9", "D9 roles", spec(D9="roles")), ("9", "D9 ages", spec(D9="ages")), ("9", "D9 basic", spec(D9="basic")),
    ("9", "switch points by ages, D9 ages", spec(r="ages", D9="ages")),
    ("9", "switch points basic, D9 ages", spec(r="basic", D9="ages")),
    ("9", "switch points basic, D9 basic", spec(r="basic", D9="basic")),
    ("9", "r7 = m1, rest ages", spec(r="ages", D9="ages", r7="const:1.0")),
    ("both", "full", spec()),
]
for who, name, sp in variants:
    t0 = time.time()
    V, R = rule_values(X, sp, co)
    sub = {5: V[5]} if who == "5" else {9: V[9]} if who == "9" else {5: V[5], 9: V[9]}
    Lr = run(X, sub=sub)[0]; r = acc_by_kind(X, Lr, KEYS)
    print(f"{('n' + who + ': ' + name)[:52]:52s}" + "  ".join(f"{100 * r[k][0]:6.1f}" for k in KEYS) +
          f"   agree {100 * (Lr.argmax(1) == L0.argmax(1)).mean():.1f}%  ({time.time() - t0:.0f}s)", flush=True)
