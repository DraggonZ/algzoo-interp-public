"""Question: which of v18's large gains does its 96% need? Start from the trained-scale build (gains 10, vote 10 per
unit; 92.1%) and set one gain back to its large default at a time: clock level g (10 -> 10,000), detector a (10 ->
1000), flag w (10 -> 10,000), tie-breaker eps (0.1 -> 1e-4 x vote ratio), and the vote scale itself (c = -9 -> 0.75,
which also scales the readout). Accuracy on 300,000 lists."""
import numpy as np
from algzoo.handbuilt_v18 import build_v18
rng = np.random.default_rng(0); X = rng.standard_normal((300000, 10)); y = np.argsort(X, 1)[:, -2]
def acc(**kw):
    P = [np.asarray(p, float) for p in build_v18(**kw).params]; h = np.zeros((len(X), 16))
    for t in range(10): h = np.maximum(h @ P[1].T + X[:, t:t + 1] * P[0], 0)
    return ((h @ P[2].T).argmax(1) == y).mean()
base = dict(g=10, a=10, w=10, lam=10.0, c=-9.0, eps=0.1)
print(f"trained scale {base}: {acc(**base):.4f}")
for k, v in [("g", 1e4), ("a", 1e3), ("w", 1e4), ("eps", 1e-3), ("g", 100), ("a", 100), ("w", 100)]:
    kw = dict(base); kw[k] = v; print(f"  {k} = {v:g}: {acc(**kw):.4f}")
kw = dict(base); kw.update(g=1e4, a=1e3); print(f"  g and a large: {acc(**kw):.4f}")
kw = dict(base); kw.update(g=1e4, a=1e3, w=1e4); print(f"  g, a, w large: {acc(**kw):.4f}")
print(f"default (all large): {acc():.4f}")
