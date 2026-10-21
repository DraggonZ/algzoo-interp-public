"""Recheck ARC's neuron 3 after review: how often does it fire, and what happens when only its final
direct output is set to its average (instead of zero), or resampled from another input?"""
import torch as th

from algzoo.core import example, forward

P = example(); wo = th.tensor(P[2])
g = th.Generator().manual_seed(41)
x = th.randn(1_000_000, 10, generator=g, dtype=th.float64)
tgt = x.argsort(-1)[:, -2]
_, H = forward(P, x, return_hidden=True)
h = H[:, -1].clone()
print("neuron 3 fires at each step (share of inputs):", [round((H[:, t, 3] > 0).double().mean().item(), 2) for t in range(1, 11)])
for lab, v in (("zero", th.zeros(len(x), dtype=x.dtype)), ("average", th.full((len(x),), h[:, 3].mean().item(), dtype=x.dtype)),
               ("resample", h[th.randperm(len(x)), 3])):
    hh = h.clone(); hh[:, 3] = v
    pred = (hh @ wo.T).argmax(-1)
    print(f"final direct output of neuron 3 set by {lab:8s}: " + " ".join(f"{s}:{(pred[tgt==s]==s).double().mean():.2f}" for s in range(10)))
print("mean final value of neuron 3 by the second's slot:", " ".join(f"{s}:{h[tgt==s,3].mean():.1f}" for s in range(10)))
