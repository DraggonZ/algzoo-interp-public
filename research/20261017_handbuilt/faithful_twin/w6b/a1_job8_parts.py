"""w6b analysis read: job 8's test (score_wide.py: nudge the hiding slots' neurons by 10% at step 5, share left at
step 6 in the mapped activity and in the readout's view) on a candidate module, with the nudge applied to subsets of
the hiding slots' neurons, to see which neurons keep a nudge. The readout view is always measured over all hiding
neurons (as the scorer does); only the nudged set changes."""
import sys, importlib.util
import numpy as np, torch
sys.path.insert(0, "/data/output/research/algzoo/20261017_handbuilt/faithful_twin")
from score_wide import load, HID
from score_twin import lists, DEV, DT

net = load(sys.argv[1])
X = lists(200_000, 7); Xp = X[X.argmax(1) <= 3][:50000]; hs = net.S(HID)
woc = net.Wt[2][:, hs] - net.Wt[2][:, hs].mean(0)
names = {k: f"slot{i}" for i in HID for k in net.sup[i]}
sets = {"all hiding": hs, **{f"slot {i} only": net.sup[i] for i in HID}}
a_, _ = net.run(Xp); A6 = net.mapped(a_[:, 6])[:, HID]
base_r = (a_[:, 6, hs] @ woc.T).norm(dim=1).clamp(min=1e-12)
for nm, ks in sets.items():
    b_, _ = net.run(Xp, nudge=(ks, 5, 0.9)); B6 = net.mapped(b_[:, 6])[:, HID]
    left = ((B6 - A6).norm(dim=1) / A6.norm(dim=1).clamp(min=1e-9)).median().item() / 0.1
    leftr = (((b_[:, 6, hs] - a_[:, 6, hs]) @ woc.T).norm(dim=1) / base_r).median().item() / 0.1
    print(f"{nm:16s} neurons {ks}: activity {left:.3f}, readout view {leftr:.3f}")
# what the readout view at step 6 is made of: norm of each slot's part
for i in HID:
    ks = net.sup[i]; part = (a_[:, 6, ks] @ (net.Wt[2][:, ks] - net.Wt[2][:, ks].mean(0)).T).norm(dim=1)
    print(f"readout view at step 6, slot {i}'s part: median norm {part.median().item():.4f} (all hiding {base_r.median().item():.4f})")
