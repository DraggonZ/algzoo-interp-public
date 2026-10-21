"""Noisy-copy benchmark for a 16-neuron 2nd-argmax model of length n (same protocol as M16,10's).

Each weight is multiplied by (1 + σ N(0,1)), σ in {0.005, 0.01, 0.02}, 32 copies each, plus the base model.
True accuracies from 2^31 samples each. Output: reports/bench_16_{n}.npz.
"""
import sys
import time

import numpy as np

from algzoo.core import accuracy, zoo

n = int(sys.argv[1])
N = 2**31
base = zoo(16, n)
models, names = [base], ["base"]
rng = np.random.default_rng(20261016 + n)
for sigma in (0.005, 0.01, 0.02):
    for k in range(32):
        models.append(tuple(w * (1 + sigma * rng.standard_normal(w.shape)) for w in base))
        names.append(f"mult{sigma}_{k}")
accs = []
t0 = time.time()
for i, m in enumerate(models):
    a, _ = accuracy(m, N, batch=2**24, seed=9000 + i)
    accs.append(a)
    print(f"{names[i]:14s} {a:.7f} ({time.time() - t0:.0f}s)", flush=True)
np.savez(f"/data/output/research/algzoo/20261016_run2/reports/bench_16_{n}.npz", names=np.array(names), acc=np.array(accs),
         n_truth=N, w_hi=np.stack([m[0] for m in models]), w_hh=np.stack([m[1] for m in models]), w_oh=np.stack([m[2] for m in models]))
