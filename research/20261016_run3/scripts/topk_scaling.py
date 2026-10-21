"""Limit for the top-k route: keep the k largest inputs exactly, set the rest to their conditional mean below the k-th.
How fast does the reduction's error fall with k, against sampling at a cost proportional to the number of placements?

Sampled (2^22 inputs per model, same inputs for all k), on the 22 trained models (ARC's model, zoo seeds 0-4, 16 new seeds).
A deterministic method on this route must handle each ordered placement of the top k at least once: 90, 720, 5,040, 30,240
and 151,200 placements for k = 2..6. Sampling's error at that many runs is given for comparison (a floor for the method's
cost, so the comparison favours the method)."""
import glob

import numpy as np
import torch as th

from algzoo.core import example, load, zoo

dev = "cuda"
N = 2 ** 22
nd = th.distributions.Normal(0.0, 1.0)
g = th.Generator(device=dev).manual_seed(2718)
X = th.randn(N, 10, device=dev, generator=g)
o = X.argsort(1); tgt = o[:, -2]; v = X.gather(1, o)
models = [("example", example())] + [(f"seed{s}", zoo(16, 10, s)) for s in range(5)]
models += [(f.rsplit("_", 1)[1].split(".")[0], load(f)) for f in sorted(glob.glob("/data/output/research/algzoo/20261016_run3/models/2nd_argmax_16_10_1073741824_*.pth"))]
KS = [2, 3, 4, 5, 6]
red = {}
for k in KS:
    mk = v[:, 10 - k]
    cm = -th.exp(nd.log_prob(mk)) / nd.cdf(mk).clamp_min(1e-12)
    Xr = cm[:, None].expand_as(X).clone(); Xr.scatter_(1, o[:, -k:], X.gather(1, o[:, -k:])); red[k] = Xr


def acc(p, inp):
    w_hi, w_hh, w_oh = (th.as_tensor(np.asarray(q), dtype=th.float32, device=dev) for q in p)
    h = th.zeros(N, 16, device=dev)
    for t in range(10):
        h = th.relu(h @ w_hh.T + inp[:, t, None] * w_hi)
    return ((h @ w_oh.T).argmax(1) == tgt).float().mean().item()


E = np.zeros((len(models), len(KS))); A = np.zeros(len(models))
for i, (name, p) in enumerate(models):
    A[i] = acc(p, X)
    for j, k in enumerate(KS):
        E[i, j] = acc(p, red[k]) - A[i]
    print(f"{name}: accuracy {A[i]:.4f}; reduction minus full (points) " + ", ".join(f"k={k} {100 * E[i, j]:+.2f}" for j, k in enumerate(KS)), flush=True)
pv = np.mean(A * (1 - A))
se = 100 * np.sqrt(0.5 / N * pv)   # sampling noise in each difference, roughly (same inputs, correlated)
for j, k in enumerate(KS):
    P = int(np.prod(range(10, 10 - k, -1)))
    print(f"k={k}: {P} placements; reduction RMSE over {len(models)} models {100 * np.sqrt(np.mean(E[:, j] ** 2)):.2f} points (6 original {100 * np.sqrt(np.mean(E[:6, j] ** 2)):.2f}, "
          f"16 new {100 * np.sqrt(np.mean(E[6:, j] ** 2)):.2f}); sampling at {P} runs {100 * np.sqrt(pv / P):.2f}")
