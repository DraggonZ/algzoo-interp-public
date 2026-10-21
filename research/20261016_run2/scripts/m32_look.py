"""Quick look at M32,10 (largest 2nd-argmax RNN): accuracy, weight sparsity, fit of comparator-type features.

Features at step t (after inputs x_0..x_{t-1}), all fitted as a * feature (no intercept), R^2 against activation:
  max_all  = max(0, x_0..x_{t-1})                         (running-max store, like n2)
  cmp_k    = max(0, inputs except x_{t-1-k}) - x_{t-1}     for k = 0..t-1 with k=0 meaning "max(0, all) - newest"
  jump     = max(0, x_{t-1} - max(0, x_0..x_{t-2}))        (new-max jump, like n4)
Each neuron's best feature is chosen over all steps 5..10 pooled.
"""
import sys

import numpy as np
import torch as th

from algzoo.core import accuracy, forward, load

dev = "cuda"
name = sys.argv[1] if len(sys.argv) > 1 else "2nd_argmax_32_10_1073741824_bo5.pth"
params = load(name)
w_hi, w_hh, w_oh = params
d = w_hh.shape[0]
print(name, "accuracy", accuracy(params, 10**8)[0])
# sparsity: share of each row's L1 mass in its 4 largest entries (input weight included)
rows = np.abs(np.c_[w_hh, w_hi])
top4 = np.sort(rows, 1)[:, ::-1][:, :4].sum(1) / rows.sum(1)
print("rows with >= 90% of L1 mass in 4 weights:", np.sum(top4 >= 0.9), "of", d, "; >= 80%:", np.sum(top4 >= 0.8))
with th.no_grad():
    x = th.randn(2**18, 10, device=dev, dtype=th.float64, generator=th.Generator(dev).manual_seed(0))
    _, H = forward(params, x, return_hidden=True)          # [N, 11, d]
    best = []
    for j in range(d):
        scores = {}
        for t in range(5, 11):
            h = H[:, t, j]
            xt = x[:, :t]
            newest = xt[:, -1]
            feats = {"max_all": xt.clamp(min=0).max(-1).values,
                     "jump": (newest - xt[:, :-1].clamp(min=0).max(-1).values).clamp(min=0)}
            for k in range(0, t):
                keep = [i for i in range(t) if i != t - 1 - k] if k > 0 else list(range(t))
                feats[f"cmp_{k}"] = xt[:, keep].clamp(min=0).max(-1).values - newest
            for nm, f in feats.items():
                a = (f * h).sum() / (f * f).sum()
                r2 = 1 - ((h - a * f) ** 2).sum() / ((h - h.mean()) ** 2).sum().clamp(min=1e-12)
                scores.setdefault(nm, []).append(r2.item())
        nm, sc = max(((k, np.mean(v)) for k, v in scores.items()), key=lambda z: z[1])
        best.append((j, nm, sc, top4[j], (H[:, 5:, j] > 0).double().mean().item()))
    best.sort(key=lambda z: -z[2])
    for j, nm, sc, sp, on in best:
        print(f"  n{j:2d}: best {nm:8s} R2 {sc:.3f}   top-4 weight share {sp:.2f}   active {on:.2f}")
