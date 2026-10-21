"""Exact Z_2 (exact_z2.py) against sampled Z_2 for the base model, zoo seeds and all 96 noisy copies.

Sampled Z_2: draw 10 standard Gaussians, zero all but the top two, run the model, score against
the position of the original 2nd largest. 2^27 samples per model (fresh seed per model).
Output: reports/z2_check.npz and a printed summary (z-scores of exact minus sampled).
"""
import sys
import time

import numpy as np
import torch as th

sys.path.insert(0, "/data/output/research/algzoo/20261016_m16_10/scripts")
from core import forward
from exact_z2 import exact_z2

OUT = "/data/output/research/algzoo/20261016_run2/reports/z2_check.npz"
N = 2**27
BATCH = 2**23
dev = "cuda"


@th.no_grad()
def sampled_z2(params, seed):
    g = th.Generator(device=dev).manual_seed(seed)
    hits = 0
    for _ in range(N // BATCH):
        x = th.randn(BATCH, 10, generator=g, device=dev)
        v, idx = x.topk(2, dim=-1)
        x2 = th.zeros_like(x).scatter_(1, idx, v)
        hits += (forward(params, x2).argmax(-1) == idx[:, 1]).sum().item()
    return hits / N


d = np.load("/data/output/research/algzoo/20261016_m16_10/reports/benchmark.npz")
names = d["names"]
rows = []
t0 = time.time()
for i, nm in enumerate(names):
    params = (d["w_hi"][i], d["w_hh"][i], d["w_oh"][i])
    ze, ops, nc = exact_z2(params)
    zs = sampled_z2(params, 500 + i)
    se = np.sqrt(zs * (1 - zs) / N)
    rows.append((ze, zs, se, ops, nc, d["acc"][i]))
    print(f"{nm:16s} exact {ze:.6f} sampled {zs:.6f} ± {se:.6f}  z {(ze - zs) / se:+.2f}"
          f"  cells {nc}  true acc {d['acc'][i]:.4f}  ({time.time() - t0:.0f}s)", flush=True)
rows = np.array(rows)
z = (rows[:, 0] - rows[:, 1]) / rows[:, 2]
print(f"\n{len(z)} models: mean z {z.mean():+.3f}, sd of z {z.std():.3f}, max |z| {np.abs(z).max():.2f}")
np.savez(OUT, names=names, z2_exact=rows[:, 0], z2_sampled=rows[:, 1], z2_se=rows[:, 2],
         ops=rows[:, 3], cells=rows[:, 4], acc=rows[:, 5], n_samples=N)
