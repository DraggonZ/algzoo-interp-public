"""Readable view of the switch-point tables ('ages': m1 by its age a1, m2 by its age a2 and a1) fitted in g16/g11.
Each switch point is r_j = c1(a1) m1 + c2(a1, a2) m2. Printed: c1 by a1 for r4, r7, r8, r1, r6; and c2 for r8 as a matrix
(rows a1, columns a2; upper triangle: m2 is the beaten record, a2 > a1; lower triangle: m2 a kept number, a2 < a1).
Also how often each (a1, a2) cell occurs at steps 1..8 (100,000 lists), to show which cells matter."""
import pickle
import numpy as np
from rolefeat import role_features
from gatelib import lists

co = pickle.load(open("g16_tables.pkl", "rb"))
X = lists(29, 100_000)
_, names, _ = role_features(X[:5], 5, np.zeros((5, 10), bool), groups=("m1", "m2"))
for j in (4, 7, 8, 1, 6):
    c = co[(f"r{j}", "ages")]
    print(f"r{j}: m1 coefficient by age of m1, a1 = 1..9: " + " ".join(f"{c[a]:.2f}" for a in range(9)))
c = co[("r8", "ages")]
M = np.full((10, 10), np.nan)
for k, nm in enumerate(names):
    if nm.startswith("m2"):
        a1 = int(nm.split("a1=")[1].split(",")[0]); a2 = int(nm.split("a2=")[1])
        M[a1, a2] = c[k]
cnt = np.zeros((10, 10))
for t in range(1, 9):
    _, _, info = role_features(X, t, np.zeros((len(X), 10), bool), groups=())
    ok = info["m2"] > 0
    np.add.at(cnt, (info["a1"][ok], info["a2"][ok]), 1)
print("\nr8: m2 coefficient c2(a1, a2); rows a1 = 1..8, columns a2 = 1..8 (share of steps in brackets for cells over 1%)")
tot = cnt.sum()
for a1 in range(1, 9):
    print(f"  a1={a1}: " + " ".join(("   .   " if np.isnan(M[a1, a2]) else f"{M[a1, a2]:5.2f}" + ("*" if cnt[a1, a2] / tot > 0.01 else " ") + " ") for a2 in range(1, 9)))
print("  (* = cell holds over 1% of steps with a positive second-largest)")
for j in (7, 4):
    c = co[(f"r{j}", "ages")]; M = np.full((10, 10), np.nan)
    for k, nm in enumerate(names):
        if nm.startswith("m2"):
            a1 = int(nm.split("a1=")[1].split(",")[0]); a2 = int(nm.split("a2=")[1]); M[a1, a2] = c[k]
    print(f"r{j}: m2 coefficients, median {np.nanmedian(M):.2f}, 10-90% {np.nanpercentile(M, 10):.2f}..{np.nanpercentile(M, 90):.2f}")
