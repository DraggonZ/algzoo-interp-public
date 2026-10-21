"""Which detail of the store-detector block carries the 11 points? Trained network with single changes (other weights
as trained), accuracy on 200,000 lists:
  store self-weight 0.97 -> 1 (no fade);
  detector's weight from the store -0.49 -> -0.58 (fires at the freshly stored largest, 10.16 / 17.6);
  both (the exact running maximum of idealized_ladder.py rung 1, others kept)."""
import numpy as np
from algzoo.core import example
wi0, wh0, wo0 = (np.asarray(p, float) for p in example())
rng = np.random.default_rng(24); X = rng.standard_normal((200000, 10)); y = np.argsort(X, 1)[:, -2]
def acc(wh, wi=wi0):
    h = np.zeros((len(X), 16))
    for t in range(10): h = np.maximum(h @ wh.T + X[:, t:t + 1] * wi, 0)
    return ((h @ wo0.T).argmax(1) == y).mean()
print(f"trained: {acc(wh0):.4f}   (store self {wh0[2, 2]:.3f}, store from detector {wh0[2, 4]:.3f}, detector from store {wh0[4, 2]:.3f}, detector self {wh0[4, 4]:.3f})")
for lab, ch in [("store self 1.0 (no fade)", {(2, 2): 1.0}), ("detector from store -0.577", {(4, 2): -10.16 / 17.6}),
                ("both", {(2, 2): 1.0, (4, 2): -10.16 / 17.6}), ("store self 0.985", {(2, 2): 0.985}), ("detector from store -0.52", {(4, 2): -0.52})]:
    w = wh0.copy()
    for k, v in ch.items(): w[k] = v
    print(f"{lab:28s}: {acc(w):.4f}")
