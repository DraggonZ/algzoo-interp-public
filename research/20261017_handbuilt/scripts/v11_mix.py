"""Memory gets the new largest's value wave through the late gate (as v7, after the dip); the clock hides a mix:
share beta of that value wave (late gate, stale after back-to-back new largests) and 1 - beta of the jump (detector,
one step after the new largest, wiped cleanly by a second one)."""
import numpy as np
from handbuilt_v9 import build_v7, scores
X = np.random.default_rng(30).standard_normal((200_000, 10)); B = len(X)
m1p = np.stack([np.max(np.c_[np.zeros((B, 2)), X[:, :t]], 1) for t in range(10)], 1); rec = X > m1p
o = X.argsort(1); sec, lar, thi = o[:, -2], o[:, -1], o[:, -3]
cons = (rec[:, :-1] & rec[:, 1:]).any(1); onepos = (X > 0).sum(1) <= 1
ordinary = ~rec[:, 8] & ~rec[:, 9] & ~cons & ~onepos
def show(name, **kw):
    P = scores(build_v7(**kw), X).argmax(1); ok = P == sec; bad = ~ok
    print(f"{name:30s} all {ok.mean():.3f} ord {ok[ordinary].mean():.3f} rec9 {ok[rec[:, 9]].mean():.3f} rec8 {ok[rec[:, 8]].mean():.3f} "
          f"b2b {ok[cons].mean():.3f} | wrong: third {(P == thi)[bad].mean():.2f} largest {(P == lar)[bad].mean():.2f} onepos {onepos[bad].mean():.2f}")
for beta in (1.0, 0.7, 0.5, 0.3, 0.0):
    show(f"clock value share {beta}", clock="jump", beta=beta, mem_jump=False)
