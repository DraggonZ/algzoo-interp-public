import numpy as np, sys, json
sys.path.insert(0, "scripts")
from handbuilt_v17 import build_v17, scores
np.set_printoptions(precision=2, suppress=True, linewidth=170)
kw = json.loads(sys.argv[1]); X = np.random.default_rng(30).standard_normal((20000, 10)); B = len(X)
m1p = np.stack([np.max(np.c_[np.zeros((B, 2)), X[:, :t]], 1) for t in range(10)], 1); rec = X > m1p
o = X.argsort(1); sec, lar = o[:, -2], o[:, -1]
S = scores(build_v17(**kw), X)
m = rec[np.arange(B), sec] & (lar == sec + 1) & (S.argmax(1) != sec)
for b in np.where(m)[0][:6]:
    print("x", X[b], "\n  rec", rec[b].astype(int), " second", sec[b], "named", S[b].argmax(), "\n  s", S[b])
