"""Tie-offset distance to ARC, as in typical/base_choice.py: local tie leanings per arrangement (typical/local_lean.py),
offset = -lean / slope (the relative gap at which the two answers swap), RMS over arrangements of the difference to
ARC's offsets (clipped to +-100%), and the median |lean difference| in votes. Third vs second; largest vs second for S
before L and L < 9 (as base_choice), plus, new here, largest vs second with S after L (L < 9)."""
import sys, numpy as np, torch
sys.path.insert(0, "../../family"); sys.path.insert(0, "../typical"); sys.path.insert(0, "..")
import os; os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../typical"))
from common import *
from local_lean import local
from typical import cell_info
from algzoo.handbuilt_v19 import build_v19
from build_v22a import build_v22a, LINEAR_SHAPE
L, S, Tq, valid, order = cell_info(); z = np.load("local_lean.npz")
ref = {k: (z[f"ARC_{k}_0"], z[f"ARC_{k}_2"]) for k in ("T", "L")}
C = {"v19 kappa 0.5": T(build_v19(kappa=0.5).params), "v22a linear": T(build_v22a(LINEAR_SHAPE).params), "v22a": T(build_v22a().params)}
print(f"{'network':16s} | third vs second: offset RMS diff (%), lean |diff| | largest vs second, S before L: same | largest vs second, S after L: same | median lean (votes): ARC / network, S before L; S after L")
for nm, W in C.items():
    row = []; leans = []
    for kind, sel in (("T", None), ("L", "before"), ("L", "after")):
        sol, _, N = local(W, kind, n=6000000) if sel != "after" else sol_L
        if kind == "L": sol_L = (sol, _, N)
        m = valid & (ref[kind][1] > 50) & (N > 50)
        if sel == "before": m &= (S < L) & (L < 9)
        if sel == "after": m &= (S > L) & (L < 9)
        off_a = -ref[kind][0][m, 0] / ref[kind][0][m, 1]; off_c = -sol[m, 0] / sol[m, 1]
        d = np.clip(off_c - off_a, -1, 1)
        row.append(f"{100*np.sqrt((d**2).mean()):6.2f} {np.median(np.abs(sol[m, 0] - ref[kind][0][m, 0])):6.3f}")
        if kind == "L": leans.append(f"{np.median(ref[kind][0][m, 0]):+.3f} / {np.median(sol[m, 0]):+.3f}")
    print(f"{nm:16s} | {row[0]:>16s} | {row[1]:>16s} | {row[2]:>16s} | {'; '.join(leans)}", flush=True)
