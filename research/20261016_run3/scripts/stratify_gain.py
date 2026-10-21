"""How much would the near-tie insight help an estimator that is allowed to sample? (Not allowed in ARC's challenge.)

Stratify random inputs by g = the smallest gap, over the sequence, between an arriving input and the running largest or
second (computable from the inputs without the network). With the best allocation of samples to strata (Neyman), the
variance per sample is (sum_h P_h sigma_h)^2 instead of p(1-p). The ratio is the efficiency gain.
"""
import numpy as np
import torch as th

from algzoo.bench import load_bench
from algzoo.core import example, zoo

dev = "cuda"
g = th.Generator(device=dev).manual_seed(21)
N = 2 ** 22
X = th.randn(N, 10, device=dev, generator=g)
# smallest arrival gap
gap = th.full((N,), 1e9, device=dev)
M = X[:, 0].clone(); m = th.full((N,), -1e9, device=dev)
for t in range(1, 10):
    x = X[:, t]
    gap = th.minimum(gap, th.minimum((x - M).abs(), (x - m).abs()))
    newM = th.maximum(M, x)
    m = th.where(x > M, M, th.maximum(m, x))
    M = newM
gap = gap.cpu().numpy()
neg2 = (X.sort(-1).values[:, -2] < 0).cpu().numpy()
edges = [0, 0.01, 0.03, 0.1, 0.2, 0.5, 1e9]
names, acc, models = load_bench(("mult",))
cases = [("ARC example", example())] + [(f"seed {s}", zoo(16, 10, s)) for s in range(5)] + \
        [(str(n), mm) for n, mm in zip(names, models) if str(n) in ("mult0.005_1", "mult0.01_3", "mult0.02_8")]
for lab, p in cases:
    w = [th.as_tensor(q, device=dev, dtype=th.float32) for q in p]
    h = th.zeros(N, 16, device=dev)
    for t in range(10):
        h = th.relu(h @ w[1].T + X[:, t, None] * w[0])
    right = ((h @ w[2].T).argmax(-1) == X.argsort(-1)[:, -2]).cpu().numpy().astype(float)
    pacc = right.mean()
    tot = 0.0
    rows = []
    for k in range(len(edges) - 1):
        for neg in (False, True):
            sel = (gap >= edges[k]) & (gap < edges[k + 1]) & (neg2 == neg)
            if sel.sum() == 0:
                continue
            P = sel.mean(); ph = right[sel].mean()
            tot += P * np.sqrt(ph * (1 - ph))
            rows.append((edges[k], neg, P, ph))
    gain = pacc * (1 - pacc) / tot ** 2
    print(f"{lab:12s}: accuracy {pacc:.4f}; efficiency gain from stratifying by near-tie gap and sign of the second: {gain:.1f}x")
for e0, neg, P, ph in rows[:12]:
    pass
