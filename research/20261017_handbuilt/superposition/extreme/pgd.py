"""First-order worst cases flip only 6% (radius.log): switching protects. Real search instead: projected gradient
ascent on the seven smaller numbers (top three fixed), each moved by at most eps x (second's value) and kept below the
third, 30 steps, maximizing the best wrong score minus the second's. Share of right lists flipped, by eps and by the
second-third gap. Also the same with the top three free to move (but keeping their order and roles)."""
import sys; sys.path.insert(0, "../steer")
import numpy as np
from harness import *
X0 = lists(200_000, seed=302); o = X0.argsort(1, descending=True); v = X0.sort(1, descending=True).values
ok = (run(ARC, X0).argmax(1) == o[:, 1]) & (v[:, 1] > 0.2); X0 = X0[ok]; o = o[ok]; v = v[ok]
L, S, T = o[:, 0], o[:, 1], o[:, 2]; n = torch.arange(len(X0), device=DEV)
small = torch.ones_like(X0, dtype=torch.bool); small[n, L] = small[n, S] = small[n, T] = False
gap = (v[:, 1] - v[:, 2]) / v[:, 1]; xs = X0[n, S]; xt = X0[n, T]
def attack(eps, steps=30):
    E = (eps * xs)[:, None]; Y = X0.clone()
    for _ in range(steps):
        Y = Y.detach().requires_grad_(True); sc = scores(ARC, Y); w = sc.clone(); w[n, S] = -1e9
        obj = (w.max(1).values - sc[n, S]).sum(); g, = torch.autograd.grad(obj, Y)
        Y = Y.detach() + (eps / 6) * xs[:, None] * torch.sign(g) * small
        Y = torch.max(torch.min(Y, X0 + E), X0 - E); Y = torch.where(small, torch.min(Y, (xt - 1e-6)[:, None]), X0)
    return run(ARC, Y).argmax(1) != S
print(f"{len(X0):,} right lists (second > 0.2)")
for eps in (0.01, 0.03, 0.1, 0.3):
    f = attack(eps); row = []
    for lo, hi in ((0, 0.05), (0.05, 0.1), (0.1, 0.2), (0.2, 0.5), (0.5, 1.01)):
        m = (gap >= lo) & (gap < hi); row.append(f"{lo}-{hi}: {100*f[m].double().mean().item():5.1f}%")
    print(f"smaller numbers moved by up to {eps:4.2f} x second: flipped {100*f.double().mean().item():5.1f}% | by gap " + " | ".join(row), flush=True)
