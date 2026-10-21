"""Do a network's hiding slots do the hiding? Silence the neurons of ARC's hiding slots (1, 6, 7, 8; through the map)
for one step, the step after the largest arrives; share of lists that then name the largest (WRITEUP: ARC about 70%).
Analysis only, outside the frozen scorers."""
import sys
import numpy as np, torch
sys.path.insert(0, "/data/output/research/algzoo/20261017_handbuilt/faithful_twin")
from score_wide import load, lists, HID, DEV, DT
for p in sys.argv[1:]:
    net = load(p); X = lists(400_000, 31); L = X.argmax(1); keep = L <= 8; X, L = X[keep], L[keep]
    kill = torch.zeros(len(X), 10, dtype=torch.bool, device=DEV); kill[torch.arange(len(X), device=DEV), L + 1] = True
    for name, k in (("intact", None), ("hiding slots silenced", (net.S(HID), kill))):
        Hw, _ = net.run(X, kill=k); pred = (Hw[:, -1] @ net.Wt[2].T).argmax(1)
        print(f"{p.split('/')[-1]:20s} {name:24s} names the largest {100 * (pred == L).double().mean().item():5.1f}%  right {100 * (pred == X.argsort(1, descending=True)[:, 1]).double().mean().item():5.1f}%")
