"""Intervention test of 'the erasure protects against the revealed wave's spill'. Lists: the second-largest is a record
at r, the third-largest sits at r+1 (kept), and the largest overtakes at T > r+1. Just before T, the cancelling neurons'
share of the r+1 number's vote is H_real[T-1, canc] - H_twin[T-1, canc], where the twin list has -0.5 at r+1. At step T,
after the overtaking silences them, that share is put back (cancellers + share, at least 0), and the run continues.
If the erasure protects the second, accuracy on these lists should fall. Control: the same on lists whose third sits two
places after the record (r+2), where ARC does not erase. 400,000 random lists, seed 41."""
import numpy as np
from simlib import WI, WH, WO, lists
CANC = [1, 6, 7, 8]
X = lists(41, 400_000); B = len(X); o = np.argsort(-X, 1); lar, sec, thi = o[:, 0], o[:, 1], o[:, 2]
m1p = np.stack([np.max(np.c_[np.zeros((B, 2)), X[:, :t]], 1) for t in range(10)], 1); rec = X > m1p
def first_rec_after(r):
    T = np.full(B, -1)
    for t in range(9, -1, -1):
        T = np.where(rec[:, t] & (t > r), t, T)
    return T


def run(Xs, T, share=None):
    h = np.zeros((len(Xs), 16)); hs = []
    for t in range(10):
        h = np.maximum(h @ WH.T + Xs[:, t:t + 1] * WI, 0)
        if share is not None:
            m = T == t; h[np.ix_(m, CANC)] = np.maximum(h[np.ix_(m, CANC)] + share[m], 0)
        hs.append(h.copy())
    return (h @ WO.T).argmax(1), np.stack(hs, 1)


for k in (1, 2):
    T = first_rec_after(sec)
    sel = rec[np.arange(B), sec] & (thi == sec + k) & (T > sec + k) & (lar == T) & (sec + k <= 8)
    Xs, Ts, ys = X[sel], T[sel], sec[sel]
    Xt = Xs.copy(); Xt[np.arange(len(Xs)), ys + k] = -0.5
    a0, H = run(Xs, Ts); _, Ht = run(Xt, Ts)
    share = (H - Ht)[np.arange(len(Xs)), Ts - 1][:, CANC]
    a1, _ = run(Xs, Ts, share)
    print(f"third at record + {k}: {len(Xs)} lists; accuracy normal {np.mean(a0 == ys):.3f}, erasure undone {np.mean(a1 == ys):.3f}; "
          f"answers now naming the third {np.mean(a1 == ys + k):.3f} (was {np.mean(a0 == ys + k):.3f}); share size median {np.median(np.linalg.norm(share, axis=1)):.2f}")

# control: on the record + 1 lists, put back a share of the same size in a random direction (ten draws)
T = first_rec_after(sec)
sel = rec[np.arange(B), sec] & (thi == sec + 1) & (T > sec + 1) & (lar == T) & (sec + 1 <= 8)
Xs, Ts, ys = X[sel], T[sel], sec[sel]; Xt = Xs.copy(); Xt[np.arange(len(Xs)), ys + 1] = -0.5
_, H = run(Xs, Ts); _, Ht = run(Xt, Ts); share = (H - Ht)[np.arange(len(Xs)), Ts - 1][:, CANC]
rng = np.random.default_rng(0); accs = []; thirds = []
for _ in range(10):
    d = rng.standard_normal(share.shape); d *= (np.linalg.norm(share, axis=1) / np.linalg.norm(d, axis=1))[:, None]
    a = run(Xs, Ts, d)[0]; accs.append(np.mean(a == ys)); thirds.append(np.mean(a == ys + 1))
print(f"control, record + 1 lists, same-size share in a random direction: accuracy {np.mean(accs):.3f} (range {min(accs):.3f}-{max(accs):.3f}); naming the third {np.mean(thirds):.3f}")
