"""Clock level gain g with everything else at the trained scale (a = w = 10, vote 10 per unit). Accuracy, and the share
of positive non-record numbers one step after a new largest at which the hiding group is partly off."""
import numpy as np
from algzoo.handbuilt_v18 import build_v18
rng = np.random.default_rng(0); X = rng.standard_normal((300000, 10)); y = np.argsort(X, 1)[:, -2]; N = len(X)
pm = np.maximum.accumulate(np.c_[np.zeros(N), np.maximum(X, 0)], 1); rec = X > pm[:, :-1]
after = np.zeros_like(rec); after[:, 1:] = rec[:, :-1]; m = after & (X > 0) & ~rec
for g in (10, 15, 20, 30, 50, 100, 300):
    net = build_v18(g=g, a=10, w=10, lam=10.0, c=-9.0, eps=0.1); I = net.idx; hid = [I[k] for k in ("clock0", "clock1", "clock2", "flip")]
    P = [np.asarray(p, float) for p in net.params]; h = np.zeros((N, 16)); offs = []
    for t in range(10):
        h = np.maximum(h @ P[1].T + X[:, t:t + 1] * P[0], 0); o = (h[:, hid] == 0).sum(1); offs.append((o > 0) & (o < 4))
    part = np.stack(offs, 1)[m].mean()
    print(f"g = {g:4d}: accuracy {((h @ P[2].T).argmax(1) == y).mean():.4f}; partly off one step after a record {part:.3f}")
