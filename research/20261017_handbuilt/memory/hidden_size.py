"""Question: how large is a hidden record's wave in the memory compared with the vote it reveals, in the trained network
and in the hand-built v18 (whose recirculating variant fails because the old wave stays at later positions)?
(a) record at r (2 or 5) that stays the largest: per-unit final-score change read through the memory neurons, largest
|value| over positions 0-8; (b) the same record overtaken three steps later (nudge record and overtaking number
together, as bridge/tail_cancel2.py): per-unit change at r (the revealed vote). Mean over lists.
1,500,000 lists, seed 3. v18 = algzoo.handbuilt_v18.build_v18() (gains 10,000)."""
import numpy as np
from algzoo.core import example
from algzoo.handbuilt_v18 import build_v18
X = np.random.default_rng(3).standard_normal((1_500_000, 10)); eps = 1e-3
T = [np.asarray(p, float) for p in example()]; T[0] = T[0].reshape(-1)
net = build_v18(); I = net.idx; Vp = [np.asarray(p, float) for p in net.params]; Vp[0] = Vp[0].reshape(-1)
nets = {"trained": (T, [0, 3, 10, 11, 12, 13, 14, 15]), "v18": (Vp, [I[f"mem{j}"] for j in range(8)])}
def final(P, X):
    wi, wh, wo = P; h = np.zeros((len(X), len(wi)))
    for t in range(10): h = np.maximum(h @ wh.T + X[:, t:t + 1] * wi, 0)
    return h
pm = np.maximum.accumulate(np.c_[np.zeros(len(X)), X], 1)
for r in (2, 5):
    q = r + 3
    hid = (X[:, r] > pm[:, r] + 0.05) & (X[:, r] > 0.3) & (X[:, r + 1:].max(1) < X[:, r] - 0.05)
    ov = (X[:, r] > pm[:, r] + 0.05) & (X[:, r] > 0.3) & (X[:, r + 1:q].max(1) < X[:, r]) & (X[:, q] > X[:, r] + 0.05) & (X[:, q + 1:].max(1) < X[:, q])
    for name, (P, M) in nets.items():
        wo = P[2]
        Xs = X[hid][:100_000]; Xp = Xs.copy(); Xp[:, r] += eps
        dh = (final(P, Xp) - final(P, Xs)) / eps; mem = (dh[:, M] @ wo[:, M].T).mean(0); tot = (dh @ wo.T).mean(0)
        Xs = X[ov][:100_000]; Xp = Xs.copy(); Xp[:, r] += eps; Xp[:, q] += eps
        vote = ((final(P, Xp) - final(P, Xs)) / eps @ wo.T).mean(0)[r]
        print(f"record at {r}, {name:7s}: hidden: memory share largest |value| over 0-8 {np.abs(mem[:9]).max():7.2f}, total largest |value| {np.abs(tot[:9]).max():6.2f} | overtaken at {q}: vote at r {vote:6.2f} | ratio hidden memory / vote {np.abs(mem[:9]).max() / vote:6.1f}")
