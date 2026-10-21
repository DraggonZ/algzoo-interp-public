"""Where does an earlier number's vote live at step t? Turn up different groups by 10% at the end of step 4 and compare
with the account's counterfactual (earlier number's value x 1.1). Groups: memory; memory + hiding; memory + hiding + gate;
everything except the store (2, 4) and detector (5); all 16 (exactly equal to scaling every input up to step 4)."""
import sys; sys.path.insert(0, "../../family")
import torch
from common import *
from memory_gain import run, X, l, s, t3, MEM
G = {"memory (0,3,10-15)": MEM, "+ hiding (1,6,7,8)": MEM + [1, 6, 7, 8], "+ gate (9)": MEM + [1, 6, 7, 8, 9],
     "+ detector (5)": MEM + [1, 5, 6, 7, 8, 9], "all 16": list(range(16))}
T_, e = 4, 0.10
for who, a, b in (("second early, third late", s, t3), ("third early, second late", t3, s)):
    m = (a <= T_ - 2) & (b >= T_ + 1) & (l > T_); Xm, am = X[m], a[m]
    Xc = Xm.clone(); Xc[torch.arange(len(Xm), device=DEV), am] *= (1 + e)
    base = (run(ARC, Xm) == am).double().mean().item(); cf = (run(ARC, Xc) == am).double().mean().item()
    row = [f"{k} {100*(run(ARC, Xm, T_, e, g) == am).double().mean().item():5.1f}" for k, g in G.items()]
    print(f"{who:25s}: base {100*base:5.1f}, predicted {100*cf:5.1f} | turned up: " + " | ".join(row), flush=True)
