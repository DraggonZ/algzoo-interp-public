"""When a new largest arrives, are earlier inputs' votes erased (except the old largest, which is revealed)?

On full random inputs at steps t >= 5: compare the score of the old second's slot just before and just after a new
largest arrives (after accounting for the one-slot slide), and the same for steps where the input is ordinary (below the
current second). Scores are centered; reported as the median change and the old second's rank among slots after.
"""
import numpy as np
import torch as th

from algzoo.core import example

dev = "cuda"
w = [th.as_tensor(q, device=dev, dtype=th.float32) for q in example()]
g = th.Generator(device=dev).manual_seed(31)
N = 400000
X = th.randn(N, 10, device=dev, generator=g)
h = th.zeros(N, 16, device=dev)
prev = None
out = {"new largest": [], "ordinary input": []}
rank_after = {"new largest": [], "ordinary input": []}
for t in range(10):
    h = th.relu(h @ w[1].T + X[:, t, None] * w[0])
    s = h @ w[2].T
    s = s - s.mean(-1, keepdim=True)
    if t >= 5:
        before = X[:, :t]
        srt = before.topk(2, -1)
        M, m = srt.values[:, 0], srt.values[:, 1]
        i2 = srt.indices[:, 1]
        x = X[:, t]
        idx = th.arange(N, device=dev)
        b = prev[idx, i2 + 9 - (t - 1)]
        a = s[idx, i2 + 9 - t]
        rk = (s > a[:, None]).sum(-1)
        for lab, sel in [("new largest", x > M), ("ordinary input", x < m)]:
            out[lab].append((a - b)[sel].cpu().numpy())
            rank_after[lab].append(rk[sel].cpu().numpy())
    prev = s
for lab in out:
    d = np.concatenate(out[lab]); r = np.concatenate(rank_after[lab])
    print(f"{lab}: old second's score change median {np.median(d):+.1f}; old second still the top slot after: {np.mean(r == 0):.2f}")
