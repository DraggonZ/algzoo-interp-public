"""How much of v16's error is the two-sided spill of revealed jump waves? Oracle: in score space, replace each overtaken
record's revealed jump wave A d cos(45 deg (q - r)) on memory positions 0-7 by a spike A d at r."""
import numpy as np, sys
sys.path.insert(0, "scripts")
from handbuilt_v16 import build_v16, scores
X = np.random.default_rng(30).standard_normal((200_000, 10)); B = len(X)
m1p = np.stack([np.max(np.c_[np.zeros((B, 2)), X[:, :t]], 1) for t in range(10)], 1); rec = X > m1p
o = X.argsort(1); sec, lar, thi = o[:, -2], o[:, -1], o[:, -3]
cons = (rec[:, :-1] & rec[:, 1:]).any(1); onepos = (X > 0).sum(1) <= 1
kinds = {"onepos": onepos, "last": rec[:, 9] & ~onepos, "step8": rec[:, 8] & ~rec[:, 9] & ~onepos,
         "row": cons & ~rec[:, 8] & ~rec[:, 9] & ~onepos, "ord": ~onepos & ~rec[:, 9] & ~rec[:, 8] & ~cons}
def show(name, S):
    P = S.argmax(1); ok = P == sec; bad = ~ok
    print(f"{name:34s} all {ok.mean():.3f} | " + " ".join(f"{k} {ok[m].mean():.3f}" for k, m in kinds.items()) +
          f" | wrong: third {(P == thi)[bad].mean():.2f} largest {(P == lar)[bad].mean():.2f} onepos {onepos[bad].mean():.2f}")
c = 0.75; A = 1 - c
for gamma in (0.0, 0.25):
    S = scores(build_v16(gamma=gamma), X); show(f"v16 gamma {gamma}", S)
    jump = np.where(rec, X - m1p, 0.0)
    last = 9 - np.argmax(rec[:, ::-1], 1)          # current largest's position
    q = np.arange(8)
    corr = np.zeros_like(S)
    for r in range(8):
        over = rec[:, r] & (last > r)                # overtaken record at r
        wave = A * np.cos(np.pi / 4 * (q - r)) - A * (q == r)
        corr[:, :8] -= over[:, None] * jump[:, r:r + 1] * wave
    show(f"  + oracle spike for overtaken", S + corr)
    corr2 = np.zeros_like(S)
    for r in range(8):
        over = rec[:, r] & (last > r) & (np.arange(10)[None] > r).any(1)
        ab = rec[:, r] & (last > r)
        # only the most recent overtaken record (the old largest)
        prev = np.array([np.max(np.where(rec[b, :last[b]])[0]) if rec[b, :last[b]].any() else -1 for b in range(0)]) if False else None
    print()
