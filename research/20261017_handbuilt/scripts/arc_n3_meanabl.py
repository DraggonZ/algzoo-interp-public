"""ARC: replace a neuron by its average value at each step (instead of zero), for every neuron.
Compares neuron 3 with the others."""
import numpy as np
import torch as th

from algzoo.core import example, forward

P = [th.tensor(p) for p in example()]
g = th.Generator().manual_seed(1)
x = th.randn(1_000_000, 10, generator=g, dtype=th.float64)
tgt = x.argsort(-1)[:, -2]
_, H = forward(example(), x, return_hidden=True)
mean = H[:, 1:].mean(0)  # [10, 16]


def run(x, k, mode):
    h = th.zeros(len(x), 16, dtype=x.dtype)
    for t in range(10):
        h = th.relu(h @ P[1].T + x[:, t, None] * P[0])
        if mode == "mean":
            h[:, k] = mean[t, k]
        elif mode == "zero":
            h[:, k] = 0
    return h @ P[2].T


print("neuron  zeroed   set to average")
for k in range(16):
    acc_z = (run(x, k, "zero").argmax(-1) == tgt).double().mean().item()
    acc_m = (run(x, k, "mean").argmax(-1) == tgt).double().mean().item()
    print(f"  {k:2d}    {acc_z:.3f}    {acc_m:.3f}")
