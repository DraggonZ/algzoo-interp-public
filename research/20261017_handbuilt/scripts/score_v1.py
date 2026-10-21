"""Score the hand-built network: two-spike pairs and random inputs (float64, CPU)."""
import sys

import numpy as np

from algzoo import handbuilt as hb

build = getattr(hb, sys.argv[1] if len(sys.argv) > 1 else "build_v1")
net = build()
P = net.params


def scores(x):
    return hb.run(P, x)[..., -1, :] @ P[2].T


# two-spike: largest 1 at p1, second r at p2, all else 0
for r in (0.3, 0.7, 0.95, 0.99):
    xs, tg = [], []
    for p1 in range(10):
        for p2 in range(10):
            if p1 != p2:
                x = np.zeros(10)
                x[p1], x[p2] = 1.0, r
                xs.append(x)
                tg.append(p2)
    ok = scores(np.array(xs)).argmax(-1) == np.array(tg)
    print(f"two-spike ratio {r}: {ok.sum()}/90")
rng = np.random.default_rng(12345)
N = 2_000_000
correct = 0
for _ in range(N // 200_000):
    x = rng.standard_normal((200_000, 10))
    correct += (scores(x).argmax(-1) == x.argsort(-1)[:, -2]).sum()
acc = correct / N
print(f"random inputs: {acc:.4f} (n={N}, standard error {np.sqrt(acc*(1-acc)/N):.4f})")
