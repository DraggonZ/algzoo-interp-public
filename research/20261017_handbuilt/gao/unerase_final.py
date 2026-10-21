"""Final-state version of the erasure intervention, for r+1 and r+2 erasers. Lists: the second is a record at r,
overtaken at T by the largest; the third sits at r+k. The cancelling neurons' share of the r+k number's vote if the
record had NOT been overtaken: in a matched list with the overtaking number set just below the record (record stays),
share = H[9, canc] - H'[9, canc], where H' also has -0.5 at r+k. Add that share to the cancelling neurons' final state in
the real list and read the scores. Control: a same-size share in a random direction."""
from pathlib import Path
import numpy as np
from algzoo.core import load, example
import simlib
from simlib import lists, task_vars
def net(k):
    return example() if k == "ARC" else load(str(next(Path("/data/output/research/algzoo/20261016_run3/models").glob(f"2nd_argmax_16_10_*_{k}.pth"))))
X = lists(41, 400_000); B = len(X); o = np.argsort(-X, 1); lar, sec, thi = o[:, 0], o[:, 1], o[:, 2]
m1p = np.stack([np.max(np.c_[np.zeros((B, 2)), X[:, :t]], 1) for t in range(10)], 1); rec = X > m1p
T = np.full(B, -1)
for t in range(9, -1, -1):
    T = np.where(rec[:, t] & (t > sec), t, T)
V = task_vars(X[:30_000]); Tm = np.arange(10)[None, :]; r_ = (V["newmax"] > 0) & (Tm >= 2); o_ = (V["newmax"] == 0) & (Tm >= 2)
for name, k in (("ARC", 1), ("111", 1), ("106", 2), ("102", 2)):
    P = tuple(np.asarray(p, float) for p in net(name)); P = (P[0].reshape(-1), P[1], P[2]); wo = P[2]
    H0 = simlib.run(X[:30_000], params=P)[2]
    CANC = [n for n in range(16) if (H0[:, :, n][r_] <= 1e-9).mean() >= 0.85 and (H0[:, :, n][o_] > 1e-9).mean() >= 0.5]
    sel = rec[np.arange(B), sec] & (thi == sec + k) & (T > sec + k) & (lar == T) & (sec + k <= 8)
    Xs, Ts, ys = X[sel], T[sel], sec[sel]; i = np.arange(len(Xs))
    Xn = Xs.copy(); Xn[i, Ts] = Xs[i, ys] - 0.01                      # record stays the largest
    Xn2 = Xn.copy(); Xn2[i, ys + k] = -0.5
    share = simlib.run(Xn, params=P)[2][:, 9][:, CANC] - simlib.run(Xn2, params=P)[2][:, 9][:, CANC]
    H = simlib.run(Xs, params=P)[2][:, 9]
    def answer(add):
        h = H.copy(); h[:, CANC] = np.maximum(h[:, CANC] + add, 0); return (h @ wo.T).argmax(1)
    a0, a1 = (H @ wo.T).argmax(1), answer(share)
    rng = np.random.default_rng(0); rc = []
    for _ in range(5):
        d = rng.standard_normal(share.shape); d *= (np.linalg.norm(share, axis=1) / (np.linalg.norm(d, axis=1) + 1e-12))[:, None]
        rc.append(np.mean(answer(d) == ys + k))
    print(f"{name}, third at record + {k}, cancelling neurons {CANC}: {len(Xs)} lists; third named {np.mean(a0 == ys + k):.3f} -> {np.mean(a1 == ys + k):.3f} "
          f"(random same-size share {np.mean(rc):.3f}); accuracy {np.mean(a0 == ys):.3f} -> {np.mean(a1 == ys):.3f}", flush=True)
