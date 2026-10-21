"""Short rule for the cancelling neurons: their real 4x4 block and ReLU, the real pushes from the input, the store and
the detector, and the memory's (and optionally the gates') push replaced by a table: median exact weights on the
largest so far L and the previous largest P, by step, steps since L arrived and neuron (push_roles.py; 60,000 random
lists). Before the first positive number the push is 0. Scored by ARC's mistakes reproduced."""
import numpy as np, torch
from common import *
import score
from push_roles import pushes_jac, CAN, MEM, GAT
wi, wh, wo = ARC
def tracks(X):
    n = len(X); Lp = torch.full((n,), -1, device=DEV, dtype=torch.long); Pp = Lp.clone(); m = torch.zeros(n, device=DEV, dtype=dt); out = []
    for t in range(10):
        out.append((Lp.clone(), Pp.clone()))
        new = X[:, t] > m; Pp = torch.where(new, Lp, Pp); Lp = torch.where(new, t, Lp); m = torch.where(new, X[:, t], m)
    return out
def table(src, n=60000):
    X = torch.randn(n, 10, device=DEV, dtype=dt, generator=torch.Generator(device=DEV).manual_seed(5)); _, J = pushes_jac(X, src)
    ar = torch.arange(n, device=DEV); TL = torch.zeros(10, 11, 4, device=DEV, dtype=dt); TP = torch.zeros_like(TL)
    for t, (L, P) in enumerate(tracks(X)):
        if t == 0: continue
        ok = L >= 0; age = t - L
        for a in range(1, t + 1):
            mm = ok & (age == a)
            if mm.sum() < 20: continue
            TL[t, a] = J[ar, t, :, L.clamp(min=0)][mm].median(0).values
            pm = mm & (P >= 0)
            if pm.sum() >= 20: TP[t, a] = J[ar, t, :, P.clamp(min=0)][pm].median(0).values
    return TL, TP
def runner(rule_src):
    tabs = {k: table(src) for k, src in (("memory", MEM), ("gates", GAT)) if k in rule_src}
    def f(X):
        h = torch.zeros(len(X), 16, device=DEV, dtype=dt); tr = tracks(X); ar = torch.arange(len(X), device=DEV)
        for t in range(10):
            pre = h @ wh.T + X[:, t:t + 1] * wi; L, P = tr[t]; ok = L >= 0; age = (t - L).clamp(min=0, max=10)
            xL = torch.where(ok, X[ar, L.clamp(min=0)], 0); xP = torch.where(P >= 0, X[ar, P.clamp(min=0)], 0)
            for k, src in (("memory", MEM), ("gates", GAT)):
                if k in tabs:
                    TL, TP = tabs[k]
                    pre[:, CAN] += -(h[:, src] @ wh[CAN][:, src].T) + TL[t, age] * xL[:, None] + TP[t, age] * xP[:, None]
            h = torch.relu(pre)
        return h @ wo.T
    return f
if __name__ == "__main__":
    models = {"memory push by L,P table": [runner({"memory"})], "gates push by L,P table": [runner({"gates"})],
              "both by L,P tables": [runner({"memory", "gates"})]}
    score.report(models, family=False)
