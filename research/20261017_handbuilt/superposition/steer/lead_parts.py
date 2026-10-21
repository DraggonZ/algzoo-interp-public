"""Why does turning the memory up make the earlier contender lose? Exact first-order bookkeeping on each list: the final
lead D = score(earlier contender) - score(later contender); its gradient with respect to the state at the end of step t,
times the part of that state each input put there. This splits D's dependence on the state at step t into
(group of neurons) x (input it came from). Scaling a group by (1+e) changes D by about e times that group's row."""
import sys; sys.path.insert(0, "../../family")
import numpy as np, torch
from common import *
from memory_gain import X, l, s, t3, MEM
GR = {"store 2,4": [2, 4], "memory": MEM, "hiding 1,6,7,8": [1, 6, 7, 8], "detector 5": [5], "gate 9": [9]}
def parts(W, Xm, a, b, lm, T_):
    Xm = Xm.clone().requires_grad_(True); h = torch.zeros(len(Xm), 16, device=DEV, dtype=dt)
    for t in range(T_ + 1): h = torch.relu(h @ W[1].T + Xm[:, t:t + 1] * W[0])
    ht = h.detach().requires_grad_(True); h2 = ht
    for t in range(T_ + 1, 10): h2 = torch.relu(h2 @ W[1].T + Xm[:, t:t + 1].detach() * W[0])
    sc = h2 @ W[2].T; n = torch.arange(len(Xm), device=DEV); D = sc[n, a] - sc[n, b]
    gD, = torch.autograd.grad(D.sum(), ht)                       # n x 16
    Jt = torch.stack([torch.autograd.grad(h[:, k].sum(), Xm, retain_graph=True)[0] for k in range(16)], 1)  # n x 16 x 10
    C = gD[:, :, None] * Jt * Xm.detach()[:, None, :]              # n x 16 neurons x 10 inputs
    out = {}
    for gname, g in GR.items():
        Cg = C[:, g].sum(1)
        out[gname] = (Cg[n, a], Cg[n, lm], Cg.sum(1) - Cg[n, a] - Cg[n, lm])
    return out, D.detach()
for T_ in (5,):
    for who, A_, B_ in (("second early, third late", s, t3), ("third early, second late", t3, s)):
        m = (l <= T_ - 3) & (A_ >= l + 2) & (A_ <= T_ - 2) & (B_ >= T_ + 1); Xm, am, bm, lm = X[m], A_[m], B_[m], l[m]
        out, D = parts(ARC, Xm[:20000], am[:20000], bm[:20000], lm[:20000], T_)
        print(f"step {T_}, {who}: final lead median {D.median():.2f}; per group, median part of the lead carried at step {T_} from: the earlier contender / the largest / the rest")
        for gname, (pa, pl, pr) in out.items():
            print(f"   {gname:16s} {pa.median():+8.2f} {pl.median():+8.2f} {pr.median():+8.2f}   (sum {(pa + pl + pr).median():+8.2f})")
