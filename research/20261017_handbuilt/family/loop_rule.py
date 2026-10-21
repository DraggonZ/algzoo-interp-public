"""The cancelling neurons and the memory as one loop (12 neurons: 1, 6, 7, 8, 0, 3, 10-15) on their real weights and
ReLUs; the pushes into the loop from the store, the detector and the gates replaced by tables of median exact weights
(on L, P, and the other numbers by step, position, steps since L and sign), as in canc_rule2.py. The rest of the
network runs on real weights. Variants: each source tabled alone, all three. Scored by ARC's mistakes reproduced."""
import numpy as np, torch
from common import *
import score
from canc_rule import tracks
wi, wh, wo = ARC
LOOP = [1, 6, 7, 8, 0, 3, 10, 11, 12, 13, 14, 15]; SRC = {"store": [2], "detector": [4], "gates": [5, 9]}
def jac_push(X, src):
    X = X.clone().requires_grad_(True); h = torch.zeros(len(X), 16, device=DEV, dtype=dt); outs = []
    for t in range(10):
        outs.append(h[:, src] @ wh[LOOP][:, src].T); h = torch.relu(h @ wh.T + X[:, t:t + 1] * wi)
    P = torch.stack(outs, 1); J = torch.zeros(len(X), 10, len(LOOP), 10, device=DEV, dtype=dt)
    for t in range(1, 10):
        for k in range(len(LOOP)):
            g, = torch.autograd.grad(P[:, t, k].sum(), X, retain_graph=True); J[:, t, k] = g
    return J
def table(src, n=60000):
    X = torch.randn(n, 10, device=DEV, dtype=dt, generator=torch.Generator(device=DEV).manual_seed(5)); J = jac_push(X, src)
    ar = torch.arange(n, device=DEV); K = len(LOOP)
    TL = torch.zeros(10, 11, K, device=DEV, dtype=dt); TP = torch.zeros_like(TL); TO = torch.zeros(10, 11, 10, 2, K, device=DEV, dtype=dt)
    for t, (L, P) in enumerate(tracks(X)):
        if t == 0: continue
        ok = L >= 0; age = t - L
        for a in range(1, t + 1):
            mm = ok & (age == a)
            if mm.sum() < 20: continue
            TL[t, a] = J[ar, t, :, L.clamp(min=0)][mm].median(0).values
            pm = mm & (P >= 0)
            if pm.sum() >= 20: TP[t, a] = J[ar, t, :, P.clamp(min=0)][pm].median(0).values
            for s in range(t):
                for pos in (0, 1):
                    om = mm & (L != s) & (P != s) & ((X[:, s] > 0) == bool(pos))
                    if om.sum() >= 20: TO[t, a, s, pos] = J[om, t, :, s].median(0).values
    return TL, TP, TO
def runner(srcs):
    tabs = {k: table(SRC[k]) for k in srcs}
    def f(X):
        h = torch.zeros(len(X), 16, device=DEV, dtype=dt); tr = tracks(X); ar = torch.arange(len(X), device=DEV)
        for t in range(10):
            pre = h @ wh.T + X[:, t:t + 1] * wi; L, P = tr[t]; ok = L >= 0; age = (t - L).clamp(min=0, max=10)
            xL = torch.where(ok, X[ar, L.clamp(min=0)], 0); xP = torch.where(P >= 0, X[ar, P.clamp(min=0)], 0)
            for k, (TL, TP, TO) in tabs.items():
                add = TL[t, age] * xL[:, None] + TP[t, age] * xP[:, None]
                for s in range(t):
                    o = ok & (L != s) & (P != s); pos = (X[:, s] > 0).long()
                    add = add + torch.where(o[:, None], TO[t, age, s, pos] * X[:, s:s + 1], 0)
                pre[:, LOOP] += -(h[:, SRC[k]] @ wh[LOOP][:, SRC[k]].T) + add
            h = torch.relu(pre)
        return h @ wo.T
    return f
if __name__ == "__main__":
    models = {"loop; store push tabled": [runner(["store"])], "loop; detector push tabled": [runner(["detector"])],
              "loop; gates push tabled": [runner(["gates"])], "loop; all three tabled": [runner(["store", "detector", "gates"])]}
    score.report(models, family=False)
