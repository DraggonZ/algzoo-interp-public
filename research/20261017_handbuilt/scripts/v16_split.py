import numpy as np, itertools, sys
from handbuilt_v16 import build_v16, scores
X = np.random.default_rng(30).standard_normal((200_000, 10)); B = len(X)
m1p = np.stack([np.max(np.c_[np.zeros((B, 2)), X[:, :t]], 1) for t in range(10)], 1); rec = X > m1p
o = X.argsort(1); sec, lar, thi = o[:, -2], o[:, -1], o[:, -3]
cons = (rec[:, :-1] & rec[:, 1:]).any(1); onepos = (X > 0).sum(1) <= 1
kinds = {"onepos": onepos, "last": rec[:, 9] & ~onepos, "step8": rec[:, 8] & ~rec[:, 9] & ~onepos,
         "row": cons & ~rec[:, 8] & ~rec[:, 9] & ~onepos, "ord": ~onepos & ~rec[:, 9] & ~rec[:, 8] & ~cons}
def show(name, **kw):
    P = scores(build_v16(**kw), X).argmax(1); ok = P == sec; bad = ~ok
    print(f"{name:28s} all {ok.mean():.3f} | " + " ".join(f"{k} {ok[m].mean():.3f}" for k, m in kinds.items()) +
          f" | wrong: third {(P == thi)[bad].mean():.2f} largest {(P == lar)[bad].mean():.2f} onepos {onepos[bad].mean():.2f}")
print("trained                      all 0.953 | onepos 0.224 last 0.944 step8 0.973 row 0.962 ord 0.962 | wrong: third 0.61 largest 0.15 onepos 0.18")
for gamma in (0.0, 0.25, 0.5, 0.75):
    show(f"gamma {gamma}", gamma=gamma)
for gamma in (0.0, 0.25, 0.5, 0.75):
    show(f"delay line, gamma {gamma}", gamma=gamma, ring=False)
