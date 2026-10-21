"""Analysis read: does Q carry the running second-largest, does the keep gate cut at 0.75 of it, and what does
dropping position 0 cost? (v27b drop0; not a scored run.)"""
import numpy as np
import algzoo.handbuilt_v23 as v23
from algzoo.handbuilt_v27b import build_v27b
from algzoo.handbuilt_v26 import build_v26

rng = np.random.default_rng(0); X = rng.standard_normal((200_000, 10)); y = np.argsort(-X, 1)[:, 1]


def states(net, X):
    wi, wh, wo = net.params; h = np.zeros((len(X), len(wi))); H = []
    for t in range(10):
        h = np.maximum(h @ wh.T + X[:, t:t + 1] * wi, 0); H.append(h)
    return np.stack(H, 1)


net = build_v27b(pos0="drop"); Q, G = net.idx["mem7"], net.idx["keep"]
H = states(net, X)
# s_{t-1}: second-largest of max(0, .) before t
s_prev = np.zeros((len(X), 10))
for t in range(2, 10):
    s_prev[:, t] = np.maximum(np.sort(X[:, :t], 1)[:, -2], 0)
Qv = H[:, :, Q] / 1e4; want = np.maximum(s_prev - X, 0)
print("Q vs g relu(s_prev - x): max abs error", np.abs(Qv - want)[:, 1:].max())
on = H[:, 1:, G] > 0; x = X[:, :9]; sp = s_prev[:, :9]
pred = x >= 0.75 * sp
pred &= x > 0
print("keep gate on == (x >= 0.75 s_prev and x > 0):", (on == pred).mean())
print("accuracy drop0:", (v23.scores(net, X).argmax(1) == y).mean(), " base r1a-3 rules:",
      (v23.scores(build_v26(), X).argmax(1) == y).mean(), " theta 0:", (v23.scores(build_v26(theta=0.0), X).argmax(1) == y).mean())
sc = v23.scores(net, X).argmax(1); ok = sc == y
print("accuracy on lists whose second is not at position 0:", ok[y != 0].mean(), " share with second at 0:", (y == 0).mean())
