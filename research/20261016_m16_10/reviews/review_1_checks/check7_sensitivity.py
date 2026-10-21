"""Single-weight sensitivity of accuracy among the six clean neurons (1, 2, 4, 6, 7, 8): +-1% multiplicative.
Same inputs for all evaluations (common random numbers), so differences are precise."""
import sys

sys.path.insert(0, "/data/output/research/algzoo/20261016_m16_10/scripts")
import numpy as np
import torch as th

from core import example, forward, target

dev = "cuda"
N = 2**21
clean = [1, 2, 4, 6, 7, 8]
with th.no_grad():
    x = th.randn(N, 10, device=dev, generator=th.Generator(dev).manual_seed(21))
    y = target(x)
    p1 = x.argmax(-1)
    order = x.argsort(-1, descending=True)
    age1, age2 = 9 - order[:, 0], 9 - order[:, 1]
    base = example()

    def acc_of(params):
        c = (forward(params, x).argmax(-1) == y).float()
        by_age2 = th.stack([c[age2 == a].mean() for a in range(10)])
        by_age1 = th.stack([c[age1 == a].mean() for a in range(10)])
        return c.mean().item(), by_age1.cpu().numpy(), by_age2.cpu().numpy()

    a0, b1, b2 = acc_of(base)
    rows = []
    for i in clean:
        for j in clean:
            for sgn in (+1, -1):
                w_hi, w_hh, w_oh = (w.copy() for w in base)
                w_hh[i, j] *= 1 + sgn * 0.01
                a, _, _ = acc_of((w_hi, w_hh, w_oh))
                rows.append((a - a0, i, j, sgn))
    rows.sort()
    print(f"base acc {a0:.4f}. Largest accuracy drops from a single +-1% change (W_hh[i,j], i <- j):")
    for d, i, j, s in rows[:10]:
        print(f"  W_hh[{i},{j}] {'+' if s > 0 else '-'}1%: {d*100:+.2f} points")
    drops = np.array([r[0] for r in rows])
    print(f"  {np.sum(drops < -0.01)} of 72 single changes cost more than 1 point; {np.sum(drops < -0.002)} cost more than 0.2 points")
    for (i, j) in [(2, 2)]:
        for f in (0.98, 0.99, 1.01, 1.02):
            w_hi, w_hh, w_oh = (w.copy() for w in base)
            w_hh[i, j] *= f
            a, a1, a2 = acc_of((w_hi, w_hh, w_oh))
            print(f"  W_hh[{i},{j}] x {f}: acc {a:.3f}; by age of max 0..9: " + " ".join(f"{v:.2f}" for v in a1)
                  + " | by age of 2nd: " + " ".join(f"{v:.2f}" for v in a2))
