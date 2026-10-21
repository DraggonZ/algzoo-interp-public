"""Is a model's tie-bias pattern stable when the largest's value and the background change?"""
import itertools

import numpy as np
import torch as th

from algzoo.core import example, forward, zoo

pl = np.array(list(itertools.permutations(range(10), 3)))
N = len(pl); r = np.arange(N)


def bias(P, m1, bg):
    base = np.full((N, 10), bg); base[r, pl[:, 0]] = m1; base[r, pl[:, 1]] = 1.0
    lo, hi = np.full(N, 0.8), np.full(N, 1.2)
    for _ in range(30):
        mid = (lo + hi) / 2
        x = base.copy(); x[r, pl[:, 2]] = mid
        s = forward(P, th.tensor(x, device="cuda")).cpu().numpy()
        tw = s[r, pl[:, 2]] > s[r, pl[:, 1]]
        hi = np.where(tw, mid, hi); lo = np.where(tw, lo, mid)
    return (lo + hi) / 2 - 1


settings = [(1.6, -0.46), (2.0, -0.46), (1.3, -0.46), (1.6, -0.2), (1.6, -0.8)]
for name, P in (("ARC", example()), ("zoo2", zoo(16, 10, 2))):
    Bs = [bias(P, *s) for s in settings]
    C = np.corrcoef(Bs)
    print(name, "correlation of its bias pattern with the first setting (largest 1.6, background -0.46):",
          " ".join(f"{s}:{C[0, i]:.2f}" for i, s in enumerate(settings) if i))

rng = np.random.default_rng(0)
B0 = bias(example(), 1.6, -0.46)
for sd in (0.001, 0.005, 0.01, 0.02):
    cs, typ = [], []
    for _ in range(6):
        Q = tuple(w * (1 + sd * rng.standard_normal(w.shape)) for w in example())
        Bq = bias(Q, 1.6, -0.46)
        cs.append(np.corrcoef(B0, Bq)[0, 1]); typ.append(np.median(np.abs(Bq - B0)))
    print(f"weights perturbed by {sd:.1%}: bias pattern correlation with ARC {np.mean(cs):.2f}; typical change {np.mean(typ):.3f} (ARC's typical |bias| {np.median(np.abs(B0)):.3f})")
