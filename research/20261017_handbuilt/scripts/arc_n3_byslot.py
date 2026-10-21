"""ARC: accuracy by the second's position, normal vs neuron 3 set to its per-step average.
Same comparison for neuron 13 (also tied to the second) as a reference."""
import torch as th

from algzoo.core import example, forward

P = [th.tensor(p) for p in example()]
g = th.Generator().manual_seed(3)
x = th.randn(1_000_000, 10, generator=g, dtype=th.float64)
tgt = x.argsort(-1)[:, -2]
_, H = forward(example(), x, return_hidden=True)
mean = H[:, 1:].mean(0)


def run(k):
    h = th.zeros(len(x), 16, dtype=x.dtype)
    for t in range(10):
        h = th.relu(h @ P[1].T + x[:, t, None] * P[0])
        if k is not None:
            h[:, k] = mean[t, k]
    return (h @ P[2].T).argmax(-1)


base, a3, a13 = run(None), run(3), run(13)
print("second at slot   normal   neuron 3 averaged   neuron 13 averaged   (where wrong answers go, n3 averaged: top slots)")
for s in range(10):
    m = tgt == s
    wrong = a3[m & (a3 != tgt)]
    top = th.bincount(wrong, minlength=10).float()
    top = (top / top.sum()).topk(2)
    print(f"      {s}          {(base[m]==s).double().mean():.3f}      {(a3[m]==s).double().mean():.3f}              {(a13[m]==s).double().mean():.3f}"
          f"        slots {top.indices.tolist()} ({top.values[0]:.0%}, {top.values[1]:.0%})")
