"""Where are the three-input reduction's mistakes? (constant background -0.46, sampled diagnostic)

Classes: second-third gap below 0.1; largest-second gap below 0.1; second below 0.3 (small); other.
Also: is the mistake decided at the arrival of the later of (second, third) / (largest, second)?
"""
import numpy as np
import torch as th

from algzoo.bench import load_bench
from algzoo.core import example, zoo

dev = "cuda"
N = 2 ** 22
g = th.Generator(device=dev).manual_seed(78)
X = th.randn(N, 10, device=dev, generator=g)
o = X.argsort(-1)
v = X.gather(1, o)
m1, m2, m3 = v[:, -1], v[:, -2], v[:, -3]
top3 = o[:, -3:]
Xr = th.full_like(X, -0.46)
Xr.scatter_(1, top3, X.gather(1, top3))
names, acc, models = load_bench(("mult",))
cases = [("example", example())] + [(f"seed{s}", zoo(16, 10, s)) for s in range(5)] + \
        [(str(n), mm) for n, mm in zip(names, models) if str(n) in ("mult0.01_3", "mult0.02_8")]
for name, p in cases:
    w = [th.as_tensor(q, device=dev, dtype=th.float32) for q in p]
    h = th.zeros(N, 16, device=dev)
    for t in range(10):
        h = th.relu(h @ w[1].T + Xr[:, t, None] * w[0])
    wrong = (h @ w[2].T).argmax(-1) != o[:, -2]
    c23 = (m2 - m3) < 0.1
    c12 = (m1 - m2) < 0.1
    small = m2 < 0.3
    other = ~(c23 | c12 | small)
    nw = wrong.sum().item()
    def sh(mask):
        return (wrong & mask).sum().item() / nw
    print(f"{name}: reduced accuracy {1 - wrong.float().mean().item():.4f}; mistakes: second-third gap < 0.1: {sh(c23):.2f}, "
          f"largest-second gap < 0.1: {sh(c12):.2f}, second < 0.3: {sh(small):.2f}, none of these: {sh(other):.2f}; "
          f"error rate when none of these: {(wrong & other).sum().item() / other.sum().item():.4f}")
