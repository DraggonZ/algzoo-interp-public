"""canc_rule.py plus the other numbers: in the memory's and gates' push tables, each number that is neither L nor P
gets a median exact weight by (step t, its position s, steps since L arrived, whether it was positive). Variants:
L,P only; + others by sign; + others by sign and whether it arrived after L. Scored by ARC's mistakes reproduced."""
import numpy as np, torch
from common import *
import score
from push_roles import pushes_jac, CAN, MEM, GAT
from canc_rule import tracks
wi, wh, wo = ARC
def table(src, others, n=80000):
    X = torch.randn(n, 10, device=DEV, dtype=dt, generator=torch.Generator(device=DEV).manual_seed(5)); _, J = pushes_jac(X, src)
    ar = torch.arange(n, device=DEV); TL = torch.zeros(10, 11, 4, device=DEV, dtype=dt); TP = torch.zeros_like(TL)
    TO = torch.zeros(10, 11, 10, 2, 4, device=DEV, dtype=dt)   # t, age, s, positive
    for t, (L, P) in enumerate(tracks(X)):
        if t == 0: continue
        ok = L >= 0; age = t - L
        for a in range(1, t + 1):
            mm = ok & (age == a)
            if mm.sum() < 20: continue
            TL[t, a] = J[ar, t, :, L.clamp(min=0)][mm].median(0).values
            pm = mm & (P >= 0)
            if pm.sum() >= 20: TP[t, a] = J[ar, t, :, P.clamp(min=0)][pm].median(0).values
            if others:
                for s in range(t):
                    for pos in (0, 1):
                        om = mm & (L != s) & (P != s) & ((X[:, s] > 0) == bool(pos))
                        if om.sum() >= 20: TO[t, a, s, pos] = J[om, t, :, s].median(0).values
    return TL, TP, TO
def runner(rule_src, others):
    tabs = {k: table(src, others) for k, src in (("memory", MEM), ("gates", GAT)) if k in rule_src}
    def f(X):
        h = torch.zeros(len(X), 16, device=DEV, dtype=dt); tr = tracks(X); ar = torch.arange(len(X), device=DEV)
        for t in range(10):
            pre = h @ wh.T + X[:, t:t + 1] * wi; L, P = tr[t]; ok = L >= 0; age = (t - L).clamp(min=0, max=10)
            xL = torch.where(ok, X[ar, L.clamp(min=0)], 0); xP = torch.where(P >= 0, X[ar, P.clamp(min=0)], 0)
            for k, src in (("memory", MEM), ("gates", GAT)):
                if k not in tabs: continue
                TL, TP, TO = tabs[k]
                add = TL[t, age] * xL[:, None] + TP[t, age] * xP[:, None]
                for s in range(t):
                    o = ok & (L != s) & (P != s); pos = (X[:, s] > 0).long()
                    add = add + torch.where(o[:, None], TO[t, age, s, pos] * X[:, s:s + 1], 0)
                pre[:, CAN] += -(h[:, src] @ wh[CAN][:, src].T) + add
            h = torch.relu(pre)
        return h @ wo.T
    return f
if __name__ == "__main__":
    models = {"memory: L,P + others by sign": [runner({"memory"}, True)], "gates: L,P + others by sign": [runner({"gates"}, True)],
              "both: L,P + others by sign": [runner({"memory", "gates"}, True)]}
    score.report(models, family=False)
