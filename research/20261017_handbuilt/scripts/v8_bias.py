"""v8 (z, no second store): memory votes are zero-mean, so memory positions score v_p - (sum of memory votes)/8 while
positions 8 and 9 score their bare vote. Check: oracle subtract (sum of memory votes)/8 from positions 8 and 9."""
import numpy as np
from handbuilt_v8 import build_v7, scores
X = np.random.default_rng(30).standard_normal((200_000, 10)); B = len(X)
m1p = np.stack([np.max(np.c_[np.zeros((B, 2)), X[:, :t]], 1) for t in range(10)], 1); rec = X > m1p
sec = X.argsort(1)[:, -2]; c = 0.75
# kept votes that sit in memory at the end: positions 0-7, positive, not a new largest, after the last new largest
# before position 8 (the dip clears older ones); plus the revealed old largest's spike
net = build_v7(clock="z", second=False); S = scores(net, X)
lastrec = np.array([np.flatnonzero(r[:8])[-1] if r[:8].any() else -1 for r in rec])
pos = np.arange(10)[None, :]
inmem = (pos < 8) & (X > 0) & ~rec & (pos > lastrec[:, None])
sumv = ((1 - c) * X * inmem).sum(1)
for k in (0, 1):
    S2 = S.copy(); S2[:, 8:] -= k * sumv[:, None] / 8
    print(f"subtract {k} x (sum of kept votes in memory)/8 from positions 8, 9: accuracy {(S2.argmax(1) == sec).mean():.3f}")
