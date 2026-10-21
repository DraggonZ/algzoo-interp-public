"""Question: at the trained network's gain scale, does v18's hiding group switch partly off the way the trained
network's does? Share of positive numbers arriving k steps after a new largest (k = 1, 2, 3; not records) at which
some but not all hiding neurons are off. Trained: neurons 1, 6, 7, 8. v18: clock 0-2 and flip, at gains 10,000 and 10."""
import numpy as np
from algzoo.core import example
from algzoo.handbuilt_v18 import build_v18
rng = np.random.default_rng(7); X = rng.standard_normal((200000, 10)); N = len(X)
def states(P, X):
    a, b, _ = (np.asarray(p, float) for p in P); h = np.zeros((len(X), len(a))); hs = []
    for t in range(10): h = np.maximum(h @ b.T + X[:, t:t + 1] * a, 0); hs.append(h)
    return np.stack(hs, 1)
pm = np.maximum.accumulate(np.c_[np.zeros(N), np.maximum(X, 0)], 1)
rec = X > pm[:, :-1]; lastrec = np.full(N, -1); since = np.zeros((N, 10), int)
for t in range(10):
    lastrec = np.where(rec[:, t], t, lastrec); since[:, t] = np.where(lastrec >= 0, t - lastrec, 99)
n18 = build_v18().idx; hid18 = [n18[k] for k in ("clock0", "clock1", "clock2", "flip")]
nets = {"trained": (example(), [1, 6, 7, 8]), "v18 gains 10,000": (build_v18().params, hid18),
        "v18 gains 10": (build_v18(g=10, a=10, w=10, lam=10.0, c=-9.0, eps=0.1).params, hid18)}
for name, (P, hid) in nets.items():
    H = states(P, X)[:, :, hid]; off = (H == 0).sum(2)
    out = []
    for k in (1, 2, 3):
        m = (since == k) & (X > 0) & ~rec
        part = ((off[m] > 0) & (off[m] < len(hid))).mean(); allo = (off[m] == len(hid)).mean()
        out.append(f"{k} step(s) after: partly off {part:.2f}, all off {allo:.2f}")
    print(f"{name:17s} " + " | ".join(out))
