"""Where do v18's small-gain losses come from? v18 at g = 10: lists with exactly one partial switch of the hiding group
(at a positive non-record number), split by how many steps after the latest record it happened; accuracy on each kind,
against lists with none."""
import numpy as np
from algzoo.handbuilt_v18 import build_v18
rng = np.random.default_rng(15); X = rng.standard_normal((600000, 10)); N = len(X); y = np.argsort(X, 1)[:, -2]
net = build_v18(g=10, a=10, w=10, lam=10.0, c=-9.0, eps=0.1); I = net.idx; hid = [I[k] for k in ("clock0", "clock1", "clock2", "flip")]
P = [np.asarray(p, float) for p in net.params]; h = np.zeros((N, 16)); parts = []
for t in range(10):
    h = np.maximum(h @ P[1].T + X[:, t:t + 1] * P[0], 0); o = (h[:, hid] == 0).sum(1); parts.append((o > 0) & (o < 4))
ok = (h @ P[2].T).argmax(1) == y; parts = np.stack(parts, 1)
pm = np.maximum.accumulate(np.c_[np.full(N, -np.inf), X], 1)[:, :-1]; rec = X > pm
last = np.full(N, -1); since = np.zeros((N, 10), int)
for t in range(10): last = np.where(rec[:, t], t, last); since[:, t] = t - last
ev = parts & (X > 0) & ~rec; n = ev.sum(1)
print(f"no partial switch: share {np.mean(n == 0):.3f}, accuracy {ok[n == 0].mean():.3f}")
for k in range(1, 6):
    sel = (n == 1) & ((ev & (since == k)).any(1))
    print(f"one partial switch, {k} step(s) after the latest record: share {sel.mean():.4f}, accuracy {ok[sel].mean():.3f}")
print(f"two or more: share {np.mean(n >= 2):.4f}, accuracy {ok[n >= 2].mean():.3f}")
# records that do not fully silence the hiding group (incomplete wipe), and lists with neither kind of event
wip = rec & ~((np.stack([None] * 0 + [0] * 0, 0) if False else np.zeros_like(rec)))
h = np.zeros((N, 16)); full = []
for t in range(10):
    h = np.maximum(h @ P[1].T + X[:, t:t + 1] * P[0], 0); full.append((h[:, hid] == 0).all(1))
full = np.stack(full, 1); inc = rec & ~full & (np.arange(10)[None, :] >= 1) & (pm > 0)
ni = inc.sum(1)
print(f"records that leave the hiding group partly on: {inc.sum() / max((rec & (pm > 0)).sum(), 1):.3f} of records after a positive number; lists with one or more: {np.mean(ni > 0):.3f}, accuracy {ok[ni > 0].mean():.3f}")
clean = (n == 0) & (ni == 0)
print(f"lists with no partial switch and no incomplete wipe: share {clean.mean():.3f}, accuracy {ok[clean].mean():.3f}")
