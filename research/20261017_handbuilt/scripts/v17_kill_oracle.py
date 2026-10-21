"""Score-space oracle on v16 (gamma 0.25): shrink the votes of kept non-records written before the current largest
(positions < L, which can never be the second) to k times their size; optionally also make every overtaken record's
reveal a spike. Votes are read from v16's keep gate one step after each number."""
import numpy as np, sys
sys.path.insert(0, "scripts")
from handbuilt_v16 import build_v16, scores
from algzoo.handbuilt import run
X = np.random.default_rng(30).standard_normal((200_000, 10)); B = len(X)
m1p = np.stack([np.max(np.c_[np.zeros((B, 2)), X[:, :t]], 1) for t in range(10)], 1); rec = X > m1p
o = X.argsort(1); sec, lar, thi = o[:, -2], o[:, -1], o[:, -3]
cons = (rec[:, :-1] & rec[:, 1:]).any(1); onepos = (X > 0).sum(1) <= 1
kinds = {"onepos": onepos, "last": rec[:, 9] & ~onepos, "step8": rec[:, 8] & ~rec[:, 9] & ~onepos,
         "row": cons & ~rec[:, 8] & ~rec[:, 9] & ~onepos, "ord": ~onepos & ~rec[:, 9] & ~rec[:, 8] & ~cons}
def show(name, S):
    P = S.argmax(1); ok = P == sec; bad = ~ok
    print(f"{name:30s} all {ok.mean():.3f} | " + " ".join(f"{k} {ok[m].mean():.3f}" for k, m in kinds.items()) +
          f" | wrong: third {(P == thi)[bad].mean():.2f} largest {(P == lar)[bad].mean():.2f} onepos {onepos[bad].mean():.2f}")
net = build_v16(gamma=0.25); S0 = scores(net, X); H = run(net.params, X); G = net.idx["keep"]
vote = H[:, 2:10, G]                                   # vote of x_q is the keep gate at step q+1 (state index q+2)
L = 9 - np.argmax(rec[:, ::-1], 1)
A = 0.25; q = np.arange(8); jump = np.where(rec, X - m1p, 0.0)
nxt = np.full((B, 10), 99)
for t in range(8, -1, -1): nxt[:, t] = np.where(rec[:, t + 1], t + 1, nxt[:, t + 1] if t + 1 < 9 else 99)
spike = np.zeros_like(S0)
for r in range(8):
    over = rec[:, r] & (nxt[:, r] < 10)
    spike[:, :8] += (over * jump[:, r])[:, None] * (A * (q == r) - A * np.cos(np.pi / 4 * (q - r)))[None]
show("v16", S0); show("v16 + spike reveal", S0 + spike)
for k in (0.5, 0.2, 0.0):
    before = (q[None] < L[:, None]) & ~rec[:, :8]
    kill = -(1 - k) * vote * before
    S = S0.copy(); S[:, :8] += kill; show(f"shrink earlier votes to {k}", S)
    S2 = S + spike; show(f"  + spike reveal", S2)
for beta in (0.5, 0.9):
    after = (q[None] > L[:, None]) & ~rec[:, :8]
    wav = np.zeros_like(S0)
    for qq in range(8):
        wav[:, :8] += (beta * vote[:, qq] * after[:, qq])[:, None] * (np.cos(np.pi / 4 * (q - qq)) - (q == qq))[None]
    before = (q[None] < L[:, None]) & ~rec[:, :8]
    S = S0 + spike; S[:, :8] += -vote * before
    show(f"kill+spike, later votes {beta} broad", S + wav)
