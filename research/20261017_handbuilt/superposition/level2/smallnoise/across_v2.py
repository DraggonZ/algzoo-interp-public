"""Revised estimate v2 (fixed 2026-10-18 after the frozen evaluation, from per-condition checks on ARC, seeds 114 and 207
only): the fourth largest exact (7 positions x 4 values), the six below it varied one at a time over 3 nodes each with
exact network runs, three leads (over the largest, the third, the fourth) Gaussian; grid split3-gl n=16.
Usage: across_v2.py set start end (sets as in across.py). Appends: index v2 share-largest share-third share-another runs."""
import sys, time, numpy as np, torch
sys.argv_saved = list(sys.argv)
from lib import *
from estimate import estimate
which, a, b = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
src = open("across.py").read().split("with open")[0]; exec(src)        # builds names / copies exactly as across.py
with open(f"across_v2_{which}.txt", "a") as f:
    for i in range(a, b):
        W = load_net(names[i]) if which == "nets" else copies[i]
        t0 = time.time(); r = estimate(W, 16, 16, "v2", 4, kind="split3-gl"); sp = r["split"] / (1 - r["acc"])
        f.write(f"{i} {r['acc']:.6f} {sp[0]:.5f} {sp[1]:.5f} {sp[2]:.5f} {r['runs']:.0f}\n"); f.flush()
        print(which, i, f"{100*r['acc']:.3f} {time.time()-t0:.0f}s", flush=True)
