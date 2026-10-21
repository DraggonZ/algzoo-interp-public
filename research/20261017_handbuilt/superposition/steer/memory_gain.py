"""Steering test. The account says the memory neurons (0, 3, 10-15) hold the votes of numbers seen earlier, while the
largest's vote is hidden elsewhere. Prediction: turning the memory up by a factor (1+e) once, at the end of step t,
should act like making the earlier numbers' values (1+e) times larger. Test on near ties between the second and third,
with the largest arriving after t (so the store is untouched): compare the intervention with the account's
counterfactual (the unmodified network on the list with the earlier number's value scaled by 1+e)."""
import sys; sys.path.insert(0, "../../family")
import numpy as np, torch
from common import *
MEM = [0, 3, 10, 11, 12, 13, 14, 15]
def run(W, X, t_int=None, e=0.0, group=MEM):
    h = torch.zeros(len(X), 16, device=DEV, dtype=dt)
    for t in range(10):
        h = torch.relu(h @ W[1].T + X[:, t:t + 1] * W[0])
        if t == t_int: h[:, group] *= (1 + e)
    return (h @ W[2].T).argmax(1)
g = torch.Generator(device=DEV).manual_seed(5); X = torch.randn(8_000_000, 10, device=DEV, dtype=dt, generator=g)
v = X.sort(1, descending=True).values; l, s, t3 = top3(X)
near = (v[:, 1] > 0) & ((v[:, 1] - v[:, 2]) < 0.05 * v[:, 1])
X, l, s, t3 = X[near], l[near], s[near], t3[near]; n = torch.arange(len(X), device=DEV)
print(f"{len(X):,} near-tie lists (third within 5% of the second, second positive)")
for T_ in (4, 6):
    for who, a, b in (("second early, third late", s, t3), ("third early, second late", t3, s)):
        m = (a <= T_ - 2) & (b >= T_ + 1) & (l > T_)      # earlier number already written; largest arrives later
        Xm, am, bm = X[m], a[m], b[m]
        base = (run(ARC, Xm) == am).double().mean().item()
        row = []
        for e in (-0.10, -0.05, 0.05, 0.10):
            iv = (run(ARC, Xm, T_, e) == am).double().mean().item()
            Xc = Xm.clone(); Xc[torch.arange(len(Xm), device=DEV), am] *= (1 + e)
            cf = (run(ARC, Xc) == am).double().mean().item()
            row.append(f"e {e:+.2f}: turned up {100*iv:5.1f} / predicted {100*cf:5.1f}")
        print(f"step {T_}, {who:25s} n {len(Xm):6,}: earlier number wins {100*base:5.1f}% | " + " | ".join(row), flush=True)
