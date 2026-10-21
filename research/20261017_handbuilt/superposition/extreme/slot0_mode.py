"""Slot 0 under-delivers when the largest arrives at step 7 or 9 (not 8, nor 1-6). Guess: the memory's rotation brings
the old vote to a bad phase then. Test: on the failing lists, remove 20% of one rotation mode from the final state
(modes.py projector). If one mode carries the failure, removing it should cut the excess errors there more than it
costs elsewhere."""
import sys; sys.path.insert(0, "../steer"); sys.path.insert(0, "../steer")
import numpy as np
from harness import *
from phase import X, S, L, T, wr
from modes import ROT, ev, R, Linv, ang
groups = {}
for i in np.argsort(ang): groups.setdefault(int(round(ang[i])), []).append(i)
def damped(Xm, sel, f):
    _, Hs = run(ARC, Xm, keep=True); h = Hs[-1].clone()
    P = torch.tensor((R[:, sel] @ Linv[sel, :]).real, device=DEV, dtype=dt); h[:, ROT] = h[:, ROT] - f * h[:, ROT] @ P.T
    return (h @ ARC[2].T).argmax(1)
for lab, m in (("second 0, largest 9", (S == 0) & (L == 9)), ("second 0, largest 7", (S == 0) & (L == 7)), ("second 1, largest 9", (S == 1) & (L == 9)), ("all clear-gap (first 2M)", torch.arange(len(X), device=DEV) < 2_000_000)):
    row = [f"base {100*wr[m].double().mean().item():.2f}%"]
    for key, sel in sorted(groups.items()):
        for f in (0.2, -0.2):
            row.append(f"{key}°{'-' if f > 0 else '+'}: {100*(damped(X[m], sel, f) != S[m]).double().mean().item():.2f}")
    print(f"{lab:26s} wrong: " + " | ".join(row), flush=True)
