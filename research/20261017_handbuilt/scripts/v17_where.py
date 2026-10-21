"""v16 failures where the second is the overtaken old largest P (at r): who wins, by offset from r."""
import numpy as np, sys, collections
sys.path.insert(0, "scripts")
from handbuilt_v16 import build_v16, scores
X = np.random.default_rng(30).standard_normal((200_000, 10)); B = len(X)
m1p = np.stack([np.max(np.c_[np.zeros((B, 2)), X[:, :t]], 1) for t in range(10)], 1); rec = X > m1p
o = X.argsort(1); sec, lar, thi = o[:, -2], o[:, -1], o[:, -3]
S = scores(build_v16(gamma=0.25), X); P = S.argmax(1); bad = P != sec
isrec2 = rec[np.arange(B), sec]                      # second is an overtaken record
print("share of lists whose second is a record:", isrec2.mean().round(3), " accuracy there:", (~bad[isrec2]).mean().round(3),
      " elsewhere:", (~bad[~isrec2]).mean().round(3))
m = bad & isrec2 & (sec < 8)
off = collections.Counter(); kind = collections.Counter()
for b in np.where(m)[0]:
    d = P[b] - sec[b]; off[d] += 1
    kind["largest" if P[b] == lar[b] else ("third" if P[b] == thi[b] else "other")] += 1
print("winner offset from the old largest (failures, second = record):", sorted(off.items()))
print(kind)
m1 = m & (P - sec == 1)
print("offset +1 failures: winner is the largest", (P == lar)[m1].mean().round(2), "| winner kept non-record", (~rec[np.arange(B), P])[m1].mean().round(2))
mb = m & (P < sec)
print("backward failures: winner is an earlier record", rec[np.arange(B), P][mb].mean().round(2))
