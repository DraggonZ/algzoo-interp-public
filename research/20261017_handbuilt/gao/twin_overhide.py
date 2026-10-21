"""A twin that hides the current largest by more than its own wave. v18's readout subtracts the clock-plus-flip wave
from positions 0-7, which hides a record's jump and leaves its position at (1 - c) x the previous largest: a near-tie
with that number's own position, broken only by eps m1. Here the readout subtracts (1 + delta) x the wave, so the
current largest's position sits delta (1 - c) x its jump lower; an overtaken record's vote comes from the memory and is
unchanged. Built at the trained network's scale (g = a = w = 10, lam 10, vote gain 10, eps 0.1). Accuracy, the gap
between the second's and the largest's scores, and accuracy with noise."""
import numpy as np
import simlib
from algzoo.handbuilt import TO_PQ
from algzoo.handbuilt_v18 import build_v18, NM, TH

X = simlib.lists(1717, 50_000); y = X.argsort(1)[:, -2]; B = len(X); r = np.arange(B); rng = np.random.default_rng(0)
KW = dict(g=10.0, a=10.0, w=10.0, lam=10.0, c=-9.0, eps=0.1, beta=0.5)


def build(delta):
    net = build_v18(**KW); wi, wh, wo = (np.asarray(p, float).copy() for p in net.params)
    A0 = 1 - KW["c"]; A = A0 / (1 + KW["beta"]); C = [4, 5, 6]; F = 3
    lev = {k: 1 / 3 for k in C}; flip = {F: 1.0, **{k: -v for k, v in lev.items()}}
    for p in range(NM):
        jp = 7 - p
        wo[p, C] += -delta * A * np.array([np.cos(TH[jp]), np.sin(TH[jp])]) @ TO_PQ
        for j, v in flip.items():
            wo[p, j] += -delta * A * (-1) ** p * v
    return wi, wh, wo


for delta in (0.0, 0.5):
    P = build(delta); L, _, H = simlib.run(X, params=P)
    o = X.argsort(1); p1, p2 = o[:, -1], o[:, -2]; m = X[r, p2] > 0.3; gap = (L[r, p2] - L[r, p1])[m]
    typ = np.array([np.median(H[:, :, n][H[:, :, n] > 0]) if (H[:, :, n] > 0).any() else 1.0 for n in range(16)])
    acc_n = []
    for s in (1e-4, 1e-3, 1e-2):
        h = np.zeros((B, 16))
        for t in range(10):
            h = np.maximum(h @ P[1].T + X[:, t:t + 1] * P[0], 0); h = np.maximum(h + s * typ * rng.standard_normal(h.shape) * (h > 0), 0)
        acc_n.append(f"{s:g}: {((h @ P[2].T).argmax(1) == y).mean():.3f}")
    print(f"delta {delta}: accuracy {(L.argmax(1) == y).mean():.3f}; second minus largest median {np.median(gap):.2f}, 10th percentile {np.percentile(gap, 10):.3f}; with noise " + ", ".join(acc_n), flush=True)
