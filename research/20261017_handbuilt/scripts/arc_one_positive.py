"""ARC on inputs with exactly one positive number: accuracy by the second's slot and by the
positive number's slot; what does it pick?"""
import numpy as np
import torch as th

from algzoo.core import example, forward

g = th.Generator(device="cuda").manual_seed(15)
x = th.randn(8_000_000, 10, generator=g, device="cuda", dtype=th.float64)
one = (x > 0).sum(-1) == 1
x = x[one]
o = x.argsort(-1); tgt, p1 = o[:, -2], o[:, -1]
pred = forward(example(), x).argmax(-1)
ok = pred == tgt
print(f"n={len(x)}, accuracy {ok.double().mean():.3f}")
print("second's slot: accuracy   | pick distribution (slot: share)")
print("  " + " ".join(f"{s}:{ok[tgt==s].double().mean():.2f}" for s in range(10)))
print("  picks: " + " ".join(f"{s}:{(pred==s).double().mean():.2f}" for s in range(10)))
print("picks the largest:", (pred == p1).double().mean().item().__round__(3))
print("accuracy by the positive number's slot: " + " ".join(f"{s}:{ok[p1==s].double().mean():.2f}" for s in range(10)))
rel = (pred - p1)
print("pick relative to the positive's slot (pick - p1): " + " ".join(f"{d}:{(rel==d).double().mean():.2f}" for d in range(-3, 4)))
