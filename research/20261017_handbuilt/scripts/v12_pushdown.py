"""v7 without its dip: the old largest is revealed as a broad wave F times a kept vote. Each position's score is then
lowered by kappa x (sum of the scores of all later positions); fairness needs kappa = F - 1. Implemented as a new
readout: wo <- (I - kappa U) wo, U strictly upper (later positions), leaving out position 9's new-largest penalty."""
import numpy as np
from handbuilt_v9 import build_v7, scores
X = np.random.default_rng(30).standard_normal((200_000, 10)); B = len(X)
m1p = np.stack([np.max(np.c_[np.zeros((B, 2)), X[:, :t]], 1) for t in range(10)], 1); rec = X > m1p
o = X.argsort(1); sec, lar, thi = o[:, -2], o[:, -1], o[:, -3]
cons = (rec[:, :-1] & rec[:, 1:]).any(1); onepos = (X > 0).sum(1) <= 1
ordinary = ~rec[:, 8] & ~rec[:, 9] & ~cons & ~onepos; secrec = rec[np.arange(B), sec]
def pushdown(net, kappa):
    wi, wh, wo = net.params; D = net.idx["detect"]
    base = wo.copy(); base[9, D] = 0.0
    U = np.triu(np.ones((10, 10)), 1)
    wo -= kappa * U @ base
    return net
def show(name, net):
    P = scores(net, X).argmax(1); ok = P == sec; bad = ~ok
    print(f"{name:36s} all {ok.mean():.3f} ord {ok[ordinary].mean():.3f} (revealed {ok[ordinary & secrec].mean():.3f}, kept {ok[ordinary & ~secrec].mean():.3f}) "
          f"rec9 {ok[rec[:, 9]].mean():.3f} b2b {ok[cons].mean():.3f} | wrong: third {(P == thi)[bad].mean():.2f} largest {(P == lar)[bad].mean():.2f}")
show("v7 late clock with dip (reference)", build_v7(clock="late"))
for F in (1.0, 2.0, 3.0, 5.0):
    show(f"no dip, wave x{F}, push-down {F - 1:.0f}", pushdown(build_v7(clock="late", w=0.0, F=F), F - 1))
