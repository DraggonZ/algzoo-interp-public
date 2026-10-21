"""Noisy copies of M16,10 plus the zoo seeds, with near-exact accuracies (2^29 samples each).

Noise: each weight is multiplied by (1 + sigma * N(0,1)), sigma in {0.005, 0.01, 0.02}, 32 models each.
Output: reports/benchmark.npz with weights and accuracies.
"""
import time

import numpy as np

from core import accuracy, example, zoo

OUT = "/data/output/research/algzoo/20261016_m16_10/reports/benchmark.npz"
N_TRUTH = 2**29

models, names = [], []
base = example()
models.append(base)
names.append("example")
for s in range(5):
    models.append(zoo(16, 10, s))
    names.append(f"zoo_seed{s}")
rng = np.random.default_rng(20261016)
for sigma in (0.005, 0.01, 0.02):
    for k in range(32):
        models.append(tuple(w * (1 + sigma * rng.standard_normal(w.shape)) for w in base))
        names.append(f"mult{sigma}_{k}")

accs = []
t0 = time.time()
for i, m in enumerate(models):
    acc, n = accuracy(m, N_TRUTH, seed=1000 + i)
    accs.append(acc)
    print(f"{names[i]:16s} acc {acc:.6f}  ({time.time() - t0:.0f}s)", flush=True)

np.savez(
    OUT,
    names=np.array(names),
    acc=np.array(accs),
    n_truth=N_TRUTH,
    w_hi=np.stack([m[0] for m in models]),
    w_hh=np.stack([m[1] for m in models]),
    w_oh=np.stack([m[2] for m in models]),
)
print("saved", OUT)
