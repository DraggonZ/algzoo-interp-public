"""ARC on random inputs: neuron 3 held at 0. Accuracy by the second's slot, and where wrong
answers go (largest's slot or another)."""
import torch as th

from algzoo.core import example

P = [th.tensor(p) for p in example()]
g = th.Generator().manual_seed(12)
x = th.randn(1_000_000, 10, generator=g, dtype=th.float64)
o = x.argsort(-1); tgt, p1 = o[:, -2], o[:, -1]


def run(kill):
    h = th.zeros(len(x), 16, dtype=x.dtype)
    for t in range(10):
        h = th.relu(h @ P[1].T + x[:, t, None] * P[0])
        if kill:
            h[:, 3] = 0
    return (h @ P[2].T).argmax(-1)


a, k = run(False), run(True)
print("second's slot  normal  neuron 3 off   wrong answers that pick the largest")
for s in range(10):
    m = tgt == s
    w = m & (k != tgt)
    print(f"     {s}        {(a[m]==s).double().mean():.3f}    {(k[m]==s).double().mean():.3f}         {(k[w]==p1[w]).double().mean():.2f}")
print(f"overall {(a==tgt).double().mean():.3f} -> {(k==tgt).double().mean():.3f}")
