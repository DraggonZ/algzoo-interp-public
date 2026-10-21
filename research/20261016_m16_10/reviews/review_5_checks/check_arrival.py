"""Review 5 check A: is a clear new second placed on top at arrival, once virtual zeros before the sequence start are excluded?

arrival_vote.py reads scores at step t with the final readout, so slots 0..8-t stand for zeros before the sequence.
A new second x < 0 can then lose to those virtual zeros. Here: restrict to x > 0, and also follow one more step.
"""
import numpy as np

from algzoo.core import example

w_hi, w_hh, w_oh = (np.asarray(p, np.float64) for p in example())
Wc = w_oh - w_oh.mean(0)
rng = np.random.default_rng(9)
N = 200000
X = rng.standard_normal((N, 10))
hh = np.zeros((N, 16))
Ss = []
for t in range(10):
    hh = np.maximum(0, hh @ w_hh.T + X[:, t, None] * w_hi)
    Ss.append(hh @ Wc.T)
rows = []
for t in range(2, 10):
    srt = np.sort(X[:, :t], 1)
    M, m = srt[:, -1], srt[:, -2]
    x = X[:, t]
    S = Ss[t]
    top = S.argmax(1)
    new2 = (x > m) & (x < M)
    # next step, if the next input does not change the top two
    if t < 9:
        nxt = X[:, t + 1]
        quiet = nxt < x
        top_next = Ss[t + 1].argmax(1)
    else:
        quiet = np.zeros(N, bool); top_next = top
    rows.append(np.c_[np.full(N, t), x, m, M, new2, top == 9, quiet, top_next == 8])
R = np.vstack(rows)
t, x, m, M, new2, top9, quiet, top8n = R.T
new2 = new2 > 0
for lab, k in [("all new seconds (as in arrival_vote)", new2),
               ("new seconds with x > 0", new2 & (x > 0)),
               ("new seconds with x < 0", new2 & (x < 0)),
               ("new seconds x>0, x-m>0.3", new2 & (x > 0) & (x - m > 0.3)),
               ("new seconds x>0, x-m in [0,0.1)", new2 & (x > 0) & (x - m < 0.1))]:
    print(f"{lab}: n {k.sum()}, newest on top at arrival {top9[k > 0].mean():.3f}")
k = new2 & (x > 0) & (x - m > 0.3) & (top9 == 0) & (quiet > 0)
k2 = new2 & (x > 0) & (x - m > 0.3) & (top9 > 0) & (quiet > 0)
print(f"clear new seconds (x>0, x-m>0.3) NOT on top at arrival, next input quiet: n {k.sum()}, on top one step later {top8n[k].mean():.3f}")
print(f"clear new seconds on top at arrival, next input quiet: n {k2.sum()}, still on top one step later {top8n[k2].mean():.3f}")
for tt in range(2, 10):
    kk = new2 & (x > 0) & (x - m > 0.3) & (t == tt)
    print(f"  t={tt}: newest on top at arrival {top9[kk].mean():.3f} (n {kk.sum()})")
