"""Score-space value of the remaining fixes on the current v17 (delay line, forward writes, value share in memory):
(a) add the missing peak A d_P at P's position when the next number is a record; (b) erase votes of kept numbers
written before the current largest (they can never be the second)."""
import numpy as np, sys
sys.path.insert(0, "scripts")
from handbuilt_v17 import build_v17, scores
from algzoo.handbuilt import run
from v17_split import X, B, rec, m1p, sec, lar, thi, kinds, onepos
KW = dict(fwd=True, dl=True, gamma=0.25, gmem=True)
def show(name, S):
    P = S.argmax(1); ok = P == sec; bad = ~ok
    print(f"{name:34s} all {ok.mean():.3f} | " + " ".join(f"{k} {ok[m].mean():.3f}" for k, m in kinds.items()) +
          f" | wrong: third {(P == thi)[bad].mean():.2f} largest {(P == lar)[bad].mean():.2f} onepos {onepos[bad].mean():.2f}")
net = build_v17(**KW); S0 = scores(net, X); H = run(net.params, X); vote = H[:, 2:10, net.idx["keep"]]
jump = np.where(rec, X - m1p, 0.0); A = 0.25; q = np.arange(8); L = 9 - np.argmax(rec[:, ::-1], 1)
peak = np.zeros_like(S0)
for r in range(8): peak[:, r] += A * jump[:, r] * rec[:, r] * rec[:, r + 1]
erase = np.zeros_like(S0); erase[:, :8] = -vote * ((q[None] < L[:, None]) & ~rec[:, :8])
show("v17 now", S0); show("+ peak when two in a row", S0 + peak); show("+ erase earlier votes", S0 + erase); show("+ both", S0 + peak + erase)
prevrec = np.zeros((B, 8), bool); prevrec[:, 1:] = rec[:, :7]
for name, mask in (("erase only right after a record", (q[None] < L[:, None]) & ~rec[:, :8] & prevrec),
                   ("erase only within the old largest's stretch", None)):
    if mask is None:
        Pp = np.array([np.max(np.where(rec[b, :L[b]])[0]) if rec[b, :L[b]].any() else -1 for b in range(B)])
        mask = (q[None] > Pp[:, None]) & (q[None] < L[:, None]) & ~rec[:, :8]
    E = np.zeros_like(S0); E[:, :8] = -vote * mask; show(name, S0 + E); show("  + peak", S0 + E + peak)
