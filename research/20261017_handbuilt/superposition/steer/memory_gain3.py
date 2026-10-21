"""Corrected design. In the first test the earlier number was always the running largest at step t (the largest arrived
later), so by the account its vote was hidden in the store and hiding neurons, not in the memory; that test probed the
wrong part. Here the largest arrives first (index <= t-3), the earlier contender (second or third) at least two steps
after it and at most t-2, the later contender after t. By the account the earlier contender's vote is then in the
memory, and the largest's hidden wave is held by the timer and cancelled by the hiding neurons."""
import sys; sys.path.insert(0, "../../family")
import torch
from common import *
from memory_gain import run, X, l, s, t3, MEM
G = {"memory": MEM, "memory + hiding": MEM + [1, 6, 7, 8], "all but store (2,4)": [n for n in range(16) if n not in (2, 4)], "all 16": list(range(16))}
for T_ in (5, 7):
    for who, a, b in (("second early, third late", s, t3), ("third early, second late", t3, s)):
        m = (l <= T_ - 3) & (a >= l + 2) & (a <= T_ - 2) & (b >= T_ + 1); Xm, am, lm = X[m], a[m], l[m]
        for e in (-0.05, 0.05):
            Xc = Xm.clone(); Xc[torch.arange(len(Xm), device=DEV), am] *= (1 + e)
            pa = lambda p: f"{100*(p == am).double().mean().item():5.1f}/{100*(p == lm).double().mean().item():4.1f}"
            row = [f"{k} {pa(run(ARC, Xm, T_, e, g))}" for k, g in G.items()]
            print(f"step {T_}, {who:25s} n {len(Xm):6,} e {e:+.2f}: earlier wins/largest wins: base {pa(run(ARC, Xm))}, predicted {pa(run(ARC, Xc))} | turned up: " + " | ".join(row), flush=True)
