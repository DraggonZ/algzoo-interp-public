"""The erasure intervention (gao/unerase.py) in other networks. For a network and offset k: lists where the second is an
overtaken record at r and the third sits at r+k; put back the network's cancelling neurons' share (identified as in
canc_across.py with the 85% cutoff) of the r+k number's vote after the overtaking. Compared with a same-size random
share. Networks: new111 (erases r+1), new102 and new106 (erase r+2), and new111 at r+2 (does not erase there)."""
from pathlib import Path
import numpy as np
from algzoo.core import load
import simlib
from simlib import lists, task_vars


def net(k):
    return load(str(next(Path("/data/output/research/algzoo/20261016_run3/models").glob(f"2nd_argmax_16_10_*_{k}.pth"))))


X = lists(41, 400_000); B = len(X); o = np.argsort(-X, 1); lar, sec, thi = o[:, 0], o[:, 1], o[:, 2]
m1p = np.stack([np.max(np.c_[np.zeros((B, 2)), X[:, :t]], 1) for t in range(10)], 1); rec = X > m1p
T = np.full(B, -1)
for t in range(9, -1, -1):
    T = np.where(rec[:, t] & (t > sec), t, T)
V = task_vars(X[:30_000]); Tm = np.arange(10)[None, :]
r_ = (V["newmax"] > 0) & (Tm >= 2); o_ = (V["newmax"] == 0) & (Tm >= 2)
for name, k in (("111", 1), ("102", 2), ("106", 2), ("111", 2)):
    wi, wh, wo = (np.asarray(p, float) for p in net(name)); wi = wi.reshape(-1)
    H0 = simlib.run(X[:30_000], params=(wi, wh, wo))[2]
    CANC = [n for n in range(16) if (H0[:, :, n][r_] <= 1e-9).mean() >= 0.85 and (H0[:, :, n][o_] > 1e-9).mean() >= 0.5]

    def run(Xs, Ts, share=None):
        h = np.zeros((len(Xs), 16)); hs = []
        for t in range(10):
            h = np.maximum(h @ wh.T + Xs[:, t:t + 1] * wi, 0)
            if share is not None:
                m = Ts == t; h[np.ix_(m, CANC)] = np.maximum(h[np.ix_(m, CANC)] + share[m], 0)
            hs.append(h.copy())
        return (h @ wo.T).argmax(1), np.stack(hs, 1)
    sel = rec[np.arange(B), sec] & (thi == sec + k) & (T > sec + k) & (lar == T) & (sec + k <= 8)
    Xs, Ts, ys = X[sel], T[sel], sec[sel]; Xt = Xs.copy(); Xt[np.arange(len(Xs)), ys + k] = -0.5
    a0, H = run(Xs, Ts); _, Ht = run(Xt, Ts); share = (H - Ht)[np.arange(len(Xs)), Ts - 1][:, CANC]
    a1, _ = run(Xs, Ts, share)
    rng = np.random.default_rng(0); rc = []
    for _ in range(5):
        d = rng.standard_normal(share.shape); d *= (np.linalg.norm(share, axis=1) / (np.linalg.norm(d, axis=1) + 1e-12))[:, None]
        rc.append(np.mean(run(Xs, Ts, d)[0] == ys + k))
    print(f"new{name}, third at record + {k}, cancelling neurons {CANC}: {len(Xs)} lists; accuracy {np.mean(a0 == ys):.3f} -> {np.mean(a1 == ys):.3f}; "
          f"third named {np.mean(a0 == ys + k):.3f} -> {np.mean(a1 == ys + k):.3f} (random same-size share: {np.mean(rc):.3f})", flush=True)
