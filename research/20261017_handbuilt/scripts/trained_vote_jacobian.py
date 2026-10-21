"""Which inputs vote in ARC's model? Gradient of (score at own slot - mean score)
w.r.t. the input, on random inputs, grouped by the input's rank at arrival
(new max / new second / below second so far) and its final rank."""
import numpy as np
import torch as th

from algzoo.core import example, forward

P = example()
g = th.Generator().manual_seed(0)
x = th.randn(20000, 10, generator=g, dtype=th.float64)
J = th.func.vmap(th.func.jacrev(lambda v: forward(P, v[None])[0]))(x).numpy()  # [b, slot, input]
own = np.einsum("bii->bi", J) - J.mean(1)  # d(score_i - mean)/dx_i
xn = x.numpy()
B, n = xn.shape
# rank at arrival among prefix
arr = np.zeros((B, n), int)
for t in range(n):
    arr[:, t] = (xn[:, :t] > xn[:, t:t + 1]).sum(1)  # 0 = new max, 1 = new second, ...
fin = (xn[:, None, :] > xn[:, :, None]).sum(2)  # final rank: 0 max, 1 second
age = n - 1 - np.arange(n)
print("median vote per unit at own slot (final rank rows, arrival rank cols)")
print("final\\arrival     new max   new 2nd   3rd+")
for fr in (0, 1, 2, 3, 4):
    row = []
    for ar in (0, 1, 2):
        m = (fin == fr) & (np.minimum(arr, 2) == ar)
        row.append(f"{np.median(own[m]):8.1f} (n={m.sum():5d})" if m.sum() > 50 else "      -          ")
    print(f"rank {fr}: ", " ".join(row))
print("\nby sign of input, final rank >=2, arrival 3rd+:")
for s, m in [("x>0", xn > 0), ("x<0", xn < 0)]:
    mm = m & (fin >= 2) & (arr >= 2)
    print(f"  {s}: median {np.median(own[mm]):.1f}")
print("\nfinal rank>=2, x>0, arrival 3rd+: median vote by age")
for a in range(10):
    mm = (xn > 0) & (fin >= 2) & (arr >= 2) & (age[None] == a)
    print(f"  age {a}: {np.median(own[mm]):6.1f}  n={mm.sum()}")
