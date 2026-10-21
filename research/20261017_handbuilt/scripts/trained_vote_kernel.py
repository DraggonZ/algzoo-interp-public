"""How does a small extra number change ARC's model's scores?

Base: largest 2.0 at p1, second 1.0 at p2, others at background b. Add a small
number e at position i (e below the second). Record score changes per slot,
relative to the mean change (only differences matter).
"""
import numpy as np
import torch as th

from algzoo.core import example, forward

P = example()
rows = []
for bg in (0.0, -0.3):
    for e in (0.3, 0.6, 0.9):
        K = np.zeros((10, 10))
        cnt = np.zeros(10)
        for p1 in range(10):
            for p2 in range(10):
                if p2 == p1:
                    continue
                x0 = np.full(10, bg)
                x0[p1], x0[p2] = 2.0, 1.0
                for i in range(10):
                    if i in (p1, p2):
                        continue
                    x1 = x0.copy()
                    x1[i] = e
                    s = forward(P, th.tensor(np.stack([x0, x1]))).numpy()
                    d = (s[1] - s[0]) / (e - bg)
                    d -= d.mean()
                    # align: kernel by slot offset k - i
                    K[i] += d
                    cnt[i] += 1
        K /= cnt[:, None]
        print(f"bg {bg} e {e}: mean score change per unit, row = position of small number, col = slot")
        print(np.array2string(K, precision=0, suppress_small=True, max_line_width=200))
