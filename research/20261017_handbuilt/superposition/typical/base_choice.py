"""Which hand-built network has typical behaviour closest to ARC's? For each candidate, local tie leanings per
arrangement (local_lean.py) for third-vs-second and largest-vs-second; distance to ARC = RMS over arrangements of the
difference in tie offsets (offset = -lean / slope: the relative gap at which the two answers swap, in %), and the
median absolute difference of leans in votes. Candidates: v7z (18), v7 (16), v18, v19 kappa 0.5 / 1 / 2, v21 (28)."""
import sys, numpy as np, torch
sys.path.insert(0, "../../family"); sys.path.insert(0, "..")
from common import *
from local_lean import local
from typical import cell_info
from baseline_v7z import nets
from algzoo.handbuilt_v18 import build_v18
from algzoo.handbuilt_v19 import build_v19
from algzoo.handbuilt_v21 import build_v21
if __name__ == "__main__":
    L, S, Tq, valid, order = cell_info(); z = np.load("local_lean.npz")
    ref = {k: (z[f"ARC_{k}_0"], z[f"ARC_{k}_2"]) for k in ("T", "L")}
    C = {**{k: v for k, v in nets().items()}, "v18 (16)": T(build_v18().params), "v19 kappa 0.5": T(build_v19(kappa=0.5).params),
         "v19 kappa 1": T(build_v19(kappa=1.0).params), "v19 kappa 2": T(build_v19(kappa=2.0).params), "v21 (28)": T(build_v21())}
    print(f"{'candidate':26s} | third vs second: offset RMS diff (%)  lean |diff| (votes) | largest vs second (S before L, L < 9): offset RMS diff  lean |diff|")
    for nm, W in C.items():
        row = []
        for kind in ("T", "L"):
            sol, _, N = local(W, kind, n=6000000)
            m = valid & (ref[kind][1] > 50) & (N > 50)
            if kind == "L": m &= (S < L) & (L < 9)
            off_a = -ref[kind][0][m, 0] / ref[kind][0][m, 1]; off_c = -sol[m, 0] / sol[m, 1]
            d = np.clip(off_c - off_a, -1, 1)
            row.append(f"{100*np.sqrt((d**2).mean()):8.2f}   {np.median(np.abs(sol[m, 0] - ref[kind][0][m, 0])):8.3f}")
        print(f"{nm:26s} | {row[0]:>38s} | {row[1]:>40s}", flush=True)
