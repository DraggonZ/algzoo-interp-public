"""Hypothesis: a near tie is decided when the later contender arrives, by the comparator neurons checking it against
the stored second-largest (neuron 8 switches near x / second-so-far = 1). Test 1: how well does each hiding neuron's
pre-activation at that step predict whether the later contender wins (AUC)? Test 2: nudging the comparators by 5% at
that step only, against the same nudge one step earlier or later: if the hypothesis holds, the arrival step dominates."""
import numpy as np
from flips import *
E, Lt = torch.minimum(S, T), torch.maximum(S, T)
m = (L < E) & (Lt >= 1)                                      # largest first, so the 'second so far' at the later arrival is the earlier contender
Xm, Em, Ltm, d0m = X[m], E[m], Lt[m], d0[m]; nn = torch.arange(len(Xm), device=DEV)
win = (d0m == Ltm).cpu().numpy()
Z = []; h = torch.zeros(len(Xm), 16, device=DEV, dtype=dt)
for t in range(10):
    z = h @ ARC[1].T + Xm[:, t:t + 1] * ARC[0]; Z.append(z); h = torch.relu(z)
Z = torch.stack(Z, 1)                                          # n x 10 x 16
za = Z[nn, Ltm]                                                # pre-activations at the later contender's arrival
def auc(score, y):
    o = np.argsort(score); r = np.empty(len(o)); r[o] = np.arange(len(o)); p = y.sum(); return (r[y].sum() - p * (p - 1) / 2) / (p * (len(y) - p))
print(f"{len(Xm):,} near ties with the largest first; later contender wins {100*win.mean():.1f}%")
for k in (1, 6, 7, 8, 5, 9):
    sc = (za[:, k] / Xm[nn, Ltm].abs()).cpu().numpy(); a = auc(sc, win)
    print(f"   neuron {k} pre-activation at the later arrival (per unit of the number): AUC for 'later wins' {max(a, 1-a):.2f}; on at arrival {100*(za[:, k] > 0).double().mean().item():.0f}%")
def nudge_at(steps_of_list, f, g):
    h = torch.zeros(len(Xm), 16, device=DEV, dtype=dt)
    for t in range(10):
        h = torch.relu(h @ ARC[1].T + Xm[:, t:t + 1] * ARC[0])
        sel = (steps_of_list == t)[:, None] & torch.isin(torch.arange(16, device=DEV), torch.tensor(g, device=DEV))[None]
        h = torch.where(sel, h * f, h)
    return (h @ ARC[2].T).argmax(1)
for lab, st in (("at the later arrival", Ltm), ("one step before", Ltm - 1), ("one step after", Ltm + 1)):
    for g in ([8], [1, 6, 7, 8]):
        ch = [100 * ((nudge_at(st, f, g) == Ltm) != (d0m == Ltm)).double().mean().item() for f in (0.95, 1.05)]
        print(f"   nudge {str(g):12s} x0.95 / x1.05 {lab:22s}: decision flips {ch[0]:4.1f}% / {ch[1]:4.1f}%")
