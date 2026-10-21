"""v14: jump-driven clock and jump waves in the memory (vote-sized, no dip); the keep gate writes the previous largest's
value at each new largest's position; the clock over-hides by a factor `hide`."""
import numpy as np
from handbuilt_v9 import build_v7, scores
X = np.random.default_rng(30).standard_normal((300_000, 10)); B = len(X)
m1p = np.stack([np.max(np.c_[np.zeros((B, 2)), X[:, :t]], 1) for t in range(10)], 1); rec = X > m1p
o = X.argsort(1); sec, lar, thi = o[:, -2], o[:, -1], o[:, -3]
cons = (rec[:, :-1] & rec[:, 1:]).any(1); onepos = (X > 0).sum(1) <= 1
kinds = {"onepos": onepos, "last step": rec[:, 9] & ~onepos, "step 8": rec[:, 8] & ~rec[:, 9] & ~onepos,
         "in a row": cons & ~rec[:, 8] & ~rec[:, 9] & ~onepos, "ordinary": ~onepos & ~rec[:, 9] & ~rec[:, 8] & ~cons}
def show(name, **kw):
    P = scores(build_v7(**kw), X).argmax(1); ok = P == sec; bad = ~ok
    print(f"{name:34s} all {ok.mean():.3f} | " + " ".join(f"{k} {ok[m].mean():.3f}" for k, m in kinds.items()) +
          f" | wrong: third {(P == thi)[bad].mean():.2f} largest {(P == lar)[bad].mean():.2f} onepos {onepos[bad].mean():.2f}")
print("trained: all 0.953 | onepos 0.224 last step 0.944 step 8 0.973 in a row 0.962 ordinary 0.962 | wrong: third 0.61 largest 0.15 onepos 0.18")
for hide in (1.0, 1.1, 1.3, 1.6):
    show(f"v14 hide {hide}", clock="jump", beta=0.0, mem_jump=True, w=0.0, lam_s=10.0, keep_rec=True, hide=hide)
