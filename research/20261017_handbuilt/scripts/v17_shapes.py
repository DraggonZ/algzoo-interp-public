"""Score-space test of reveal shapes on top of v16's real scores (gamma 0.25). For every overtaken record (at r, jump d,
overtaken at L), v16 shows A d cos(45 deg (q - r)) on memory positions q = 0..7. Replace it by:
  spike     : A d at r only
  cut       : the cosine on q <= L - 2 only (what a delay line whose newest slot copies the clock would hold)
  cut+spike : spike, plus the cosine on q < r (backward spill only)"""
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
    print(f"{name:24s} all {ok.mean():.3f} | " + " ".join(f"{k} {ok[m].mean():.3f}" for k, m in kinds.items()) +
          f" | wrong: third {(P == thi)[bad].mean():.2f} largest {(P == lar)[bad].mean():.2f} onepos {onepos[bad].mean():.2f}")
A = 0.25; q = np.arange(8)
S0 = scores(build_v16(gamma=0.25), X)
jump = np.where(rec, X - m1p, 0.0)
nxt = np.full((B, 10), 99)                            # step of the next record after each position
for t in range(8, -1, -1):
    nxt[:, t] = np.where(rec[:, t + 1], t + 1, nxt[:, t + 1] if t + 1 < 9 else 99)
show("v16", S0)
for name in ("spike", "cut", "cut+spike"):
    S = S0.copy()
    for r in range(8):
        L = nxt[:, r]; over = rec[:, r] & (L < 10)
        cosw = A * np.cos(np.pi / 4 * (q - r))
        if name == "spike": new = A * (q == r) + 0 * L[:, None]
        elif name == "cut": new = np.where(q[None] <= L[:, None] - 2, cosw[None], 0.0)
        else: new = np.where(q[None] < r, cosw[None], 0.0) + A * (q == r)
        S[:, :8] += (over * jump[:, r])[:, None] * (new - cosw[None])
    show(name, S)
for phi in (15, 22.5, 30):
    S = S0.copy(); ph = np.deg2rad(phi)
    for r in range(8):
        L = nxt[:, r]; over = rec[:, r] & (L < 10)
        cosw = A * np.cos(np.pi / 4 * (q - r)); sh = A * np.cos(np.pi / 4 * (q - r) + ph) / np.cos(ph)
        new = np.where(q[None] <= L[:, None] - 2, sh[None], 0.0)
        S[:, :8] += (over * jump[:, r])[:, None] * (new - cosw[None])
    show(f"cut, phase back {phi}", S)
for th_deg in (55, 60, 72):
    S = S0.copy(); th_ = np.deg2rad(th_deg)
    for r in range(8):
        L = nxt[:, r]; over = rec[:, r] & (L < 10)
        cosw = A * np.cos(np.pi / 4 * (q - r)); sh = A * np.cos(th_ * (q - r))
        S[:, :8] += (over * jump[:, r])[:, None] * (sh - cosw)[None]
    show(f"two-sided, {th_deg} deg per position", S)
