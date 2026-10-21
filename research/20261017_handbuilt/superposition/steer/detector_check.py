"""Prediction from detector_share.log: about half of each kept vote enters through the detector one step after the
number arrives, the later contender sends a larger share (except at step 9, which bypasses it). So turning the
detector up should help the later contender, except a contender at step 9, which should lose; turning it down the
reverse."""
import sys; sys.path.insert(0, "../../family")
import torch
from common import *
from memory_gain import X, l, s, t3
from three import run_all
late = torch.where(s > t3, s, t3)
base = run_all(ARC, X, [5], 1.0)
for a in (-0.05, 0.05):
    p = run_all(ARC, X, [5], 1 + a)
    for name, m in (("later contender at steps 1-8", late <= 8), ("later contender at step 9", late == 9)):
        print(f"detector x {1+a:.2f}, {name:30s} n {int(m.sum()):7,}: later contender wins {100*(base[m] == late[m]).double().mean().item():5.1f}% -> {100*(p[m] == late[m]).double().mean().item():5.1f}%", flush=True)
