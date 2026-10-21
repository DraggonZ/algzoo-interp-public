"""Diagnostic (oracle, not a hand-built rule): does a value-proportional margin against the current largest fix v18?
Push the final score of the current largest's position down by eta x (vote per unit A0) x the largest, for v18 at the
trained gain scale (A0 = 10) and at its default gains (A0 = 0.25). eta = 0 is v18 as built."""
import numpy as np
from algzoo.handbuilt_v18 import build_v18
rng = np.random.default_rng(0); X = rng.standard_normal((300000, 10)); N = len(X); y = np.argsort(X, 1)[:, -2]
am = X.argmax(1); m1 = np.maximum(X.max(1), 0)
for name, kw, A0 in (("gains 10", dict(g=10, a=10, w=10, lam=10.0, c=-9.0, eps=0.1), 10.0), ("gains 10,000", {}, 0.25)):
    P = [np.asarray(p, float) for p in build_v18(**kw).params]; h = np.zeros((N, 16))
    for t in range(10): h = np.maximum(h @ P[1].T + X[:, t:t + 1] * P[0], 0)
    L = h @ P[2].T
    out = []
    for eta in (0.0, 0.05, 0.1, 0.2, 0.3):
        Lm = L.copy(); Lm[np.arange(N), am] -= eta * A0 * m1; out.append(f"eta {eta}: {(Lm.argmax(1) == y).mean():.4f}")
    print(f"v18 {name}: " + ", ".join(out))
