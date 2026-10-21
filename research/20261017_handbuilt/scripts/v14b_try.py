"""v14b: clock gets the jump one step after a new largest (detector); memory gets the same jump wave one step later
through the late gate (a delayed copy of the detector), after the dip; the keep gate writes the previous largest's value
at each new largest's position. Wave size F x ab, dip w."""
import numpy as np, itertools
from handbuilt_v9 import build_v7, scores
X = np.random.default_rng(30).standard_normal((200_000, 10)); B = len(X)
m1p = np.stack([np.max(np.c_[np.zeros((B, 2)), X[:, :t]], 1) for t in range(10)], 1); rec = X > m1p
o = X.argsort(1); sec, lar, thi = o[:, -2], o[:, -1], o[:, -3]
cons = (rec[:, :-1] & rec[:, 1:]).any(1); onepos = (X > 0).sum(1) <= 1
kinds = {"onepos": onepos, "last": rec[:, 9] & ~onepos, "step8": rec[:, 8] & ~rec[:, 9] & ~onepos,
         "row": cons & ~rec[:, 8] & ~rec[:, 9] & ~onepos, "ord": ~onepos & ~rec[:, 9] & ~rec[:, 8] & ~cons}
res = []
for w, F, hide in itertools.product((0.5, 0.7, 0.8, 0.9), (0.2, 0.5, 1.0), (1.0, 1.2)):
    net = build_v7(clock="jump", beta=0.0, mem_jump=False, late_jump=True, keep_rec=True, w=w, F=F, hide=hide)
    P = scores(net, X).argmax(1); ok = P == sec; bad = ~ok
    res.append((ok.mean(), w, F, hide, *[ok[m].mean() for m in kinds.values()], (P == thi)[bad].mean(), (P == lar)[bad].mean()))
res.sort(reverse=True)
print("trained: all 0.953 | onepos 0.224 last 0.944 step8 0.973 row 0.962 ord 0.962 | wrong third 0.61 largest 0.15")
for r in res[:10]:
    print("all {:.3f}  w {} F {} hide {} | onepos {:.3f} last {:.3f} step8 {:.3f} row {:.3f} ord {:.3f} | wrong third {:.2f} largest {:.2f}".format(*r))
