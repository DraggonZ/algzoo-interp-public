"""Exact per-number weights of the pushes into the cancelling neurons (1, 6, 7, 8) from the memory and from the gates.
On each list the network is linear, so push_n(t) = sum_s w[n, t, s] x_s. Group the weights by role: the largest so far
L, the previous largest P, the number just before this step (if not L), other numbers since L arrived, other numbers
before L. Report median weights by steps since L arrived (age), and how much of the push's variance a table of the
median L and P weights (by age, step and neuron) explains. Random lists."""
import numpy as np, torch
from common import *
wi, wh, wo = ARC
CAN = [1, 6, 7, 8]; MEM = [0, 3, 10, 11, 12, 13, 14, 15]; GAT = [5, 9]
def pushes_jac(X, src):
    X = X.clone().requires_grad_(True); h = torch.zeros(len(X), 16, device=DEV, dtype=dt); outs = []
    for t in range(10):
        outs.append(h[:, src] @ wh[CAN][:, src].T)   # push into cancellers at step t (from h(t-1))
        h = torch.relu(h @ wh.T + X[:, t:t + 1] * wi)
    P = torch.stack(outs, 1)   # n, t, 4
    J = torch.zeros(len(X), 10, 4, 10, device=DEV, dtype=dt)
    for t in range(1, 10):
        for k in range(4):
            g, = torch.autograd.grad(P[:, t, k].sum(), X, retain_graph=True); J[:, t, k] = g
    return P.detach(), J
if __name__ == "__main__":
    X = torch.randn(60000, 10, device=DEV, dtype=dt, generator=torch.Generator(device=DEV).manual_seed(5))
    # running largest, previous largest, ages
    n = len(X); Lp = torch.full((n,), -1, device=DEV, dtype=torch.long); Pp = torch.full((n,), -1, device=DEV, dtype=torch.long)
    m = torch.zeros(n, device=DEV, dtype=dt); LP = []
    for t in range(10):
        LP.append((Lp.clone(), Pp.clone()))
        new = X[:, t] > m; Pp = torch.where(new, Lp, Pp); Lp = torch.where(new, t, Lp); m = torch.where(new, X[:, t], m)
    ar = torch.arange(n, device=DEV)
    for nm, src in (("memory", MEM), ("gates", GAT)):
        Pu, J = pushes_jac(X, src); print(f"== push from the {nm}")
        for k, cn in enumerate(CAN):
            rows = []; tab_pred = torch.zeros(n, 10, device=DEV, dtype=dt)
            for t in range(1, 10):
                L, P = LP[t]; ok = L >= 0; age = t - L
                wL = torch.where(ok, J[ar, t, k, L.clamp(min=0)], 0); wP = torch.where(P >= 0, J[ar, t, k, P.clamp(min=0)], 0)
                xL = torch.where(ok, X[ar, L.clamp(min=0)], 0); xP = torch.where(P >= 0, X[ar, P.clamp(min=0)], 0)
                for a in range(1, t + 1):
                    mm = ok & (age == a)
                    if mm.sum() < 50: continue
                    mL, mP = wL[mm].median(), wP[mm].median(); tab_pred[mm, t] = mL * xL[mm] + mP * xP[mm]
                    if t == 6 or a in (1, 2): rows.append(f"t{t} age{a}: L {mL:+6.2f} P {mP:+6.2f}")
            v = Pu[:, 1:, k]; r = v - tab_pred[:, 1:]
            print(f"  neuron {cn}: push RMS {v.pow(2).mean().sqrt():.2f}; L,P table explains {100*(1 - r.var()/v.var()):.0f}% of its variance | " + "; ".join(rows[:6]))
