"""w6b analysis read: is build_w6b the same function as w5b_slot7 (answers and score differences on 400,000 lists), and
own accuracy of both. Also checks H7 - C7 = c x (jump of a record one step before) on those lists."""
import numpy as np
from algzoo.handbuilt_w5b import build_w5b
from algzoo.handbuilt_w6b import build_w6b

rng = np.random.default_rng(11); X = rng.standard_normal((400_000, 10)); y = np.argsort(-X, 1)[:, 1]
def run(W, keep=False):
    wi, wh, wo = W[:3]; h = np.zeros((len(X), len(wi))); hs = []
    for t in range(10):
        h = np.maximum(h @ wh.T + X[:, t:t + 1] * wi, 0); hs.append(h)
    return h @ wo.T, np.stack(hs, 1)
A, _ = run(build_w5b()); W6 = build_w6b(); B, Hs = run(W6)
idx = W6[3]
print(f"w5b_slot7 accuracy {100 * (A.argmax(1) == y).mean():.2f}%, w6b {100 * (B.argmax(1) == y).mean():.2f}%")
print(f"answers differ on {100 * (A.argmax(1) != B.argmax(1)).mean():.3f}% of lists; max |score difference| {np.abs(A - B).max():.2e}, median {np.median(np.abs(A - B)):.2e}")
m = np.maximum.accumulate(X, 1); d = np.zeros_like(X); d[:, 1:] = np.maximum(X[:, 1:] - m[:, :-1], 0); d[:, 0] = np.maximum(X[:, 0], 0)
rel = Hs[:, 1:, idx["h7"]] - Hs[:, 1:, idx["c7"]]
rec_next = X[:, 1:] > m[:, :-1]
ok = ~rec_next
print(f"H7 - C7 against the previous step's jump (steps 1-9, no record now): max error {np.abs(rel[ok] - d[:, :-1][ok]).max():.2e}; when a record arrives: max {np.abs(rel[rec_next]).max():.2e}")
