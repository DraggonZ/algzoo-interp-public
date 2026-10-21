"""Analysis: job 11's regression on each delay cell's pre-activation (largest, second, third, const), per step and age.
Usage: a1_gapcell.py <delta>   (builds v26 in v23's own neuron order)."""
import sys
import numpy as np
from algzoo.handbuilt_v26 import build_v26

delta = float(sys.argv[1])
net = build_v26(delta=delta)
wi, wh, wo = [np.asarray(p, float) for p in net.params]
rng = np.random.default_rng(7); X = rng.standard_normal((200_000, 10))
h = np.zeros((len(X), 16)); P = []
for t in range(10):
    p = h @ wh.T + X[:, t:t + 1] * wi; h = np.maximum(p, 0); P.append(p)
P = np.stack(P, 1)
am = np.zeros((len(X), 10), int); cur = np.zeros(len(X), int)
for t in range(1, 10):
    cur = np.where(X[:, t] > X[np.arange(len(X)), cur], t, cur); am[:, t] = t - cur
print("wave f(k), k=-8..8:", end=" ")
for j in range(8):
    y = net.idx[f"mem{j}"]; rows = []
    for t in range(3, 10):
        v = -np.sort(-X[:, :t + 1], 1)[:, :3]
        for a0 in (1, 2):
            m = (am[:, t] == a0) & (v[:, 0] > 0)
            A = np.c_[v[m], np.ones(m.sum())]; cf, *_ = np.linalg.lstsq(A, P[m, t, y], rcond=None)
            gp = v[m, 0] - v[m, 1]; r2 = np.corrcoef(gp, P[m, t, y])[0, 1] ** 2
            rows.append((t, a0, cf[0], cf[1], r2))
    ok = np.mean([r[2] > 0 and r[3] < 0 for r in rows]); r2m = np.median([r[4] for r in rows])
    print(f"\nmem{j}: signs {ok:.2f} R2 {r2m:.3f}  " + " ".join(f"t{t}a{a}:{c1:+.2f}/{c2:+.2f}" for t, a, c1, c2, _ in rows))
S = h @ wo.T
print(f"\naccuracy on these 200,000 lists: {(S.argmax(1) == np.argsort(-X, 1)[:, 1]).mean():.4f}")
