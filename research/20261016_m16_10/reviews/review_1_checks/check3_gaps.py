"""Are errors 'fuzzy comparisons'? Accuracy by the smallest arrival-time comparison gap.

At each step t >= 1 the ideal algorithm compares x_t with the running max m1 and running 2nd m2.
g = min over steps of |x_t - m1_t| and |x_t - m2_t|. Also the final gaps m1-m2 and m2-m3.
"""
import sys

sys.path.insert(0, "/data/output/research/algzoo/20261016_m16_10/scripts")
import numpy as np
import torch as th

from core import example, forward, target
from bench import load_bench

dev = "cuda"
N = 2**22


def gaps(x):
    n = x.shape[1]
    m1 = x[:, 0].clone()
    m2 = th.full_like(m1, -th.inf)
    g = th.full_like(m1, th.inf)
    for t in range(1, n):
        xt = x[:, t]
        g = th.minimum(g, (xt - m1).abs())
        g = th.minimum(g, th.where(th.isfinite(m2), (xt - m2).abs(), th.full_like(g, th.inf)))
        new_m1 = th.maximum(m1, xt)
        m2 = th.where(xt > m1, m1, th.maximum(m2, xt))
        m1 = new_m1
    return g


def run(params, label):
    g_ = th.Generator(dev).manual_seed(5)
    x = th.randn(N, 10, device=dev, generator=g_)
    c = (forward(params, x).argmax(-1) == target(x))
    g = gaps(x)
    xs = x.sort(-1, descending=True).values
    fg = th.minimum(xs[:, 0] - xs[:, 1], xs[:, 1] - xs[:, 2])
    edges = [0, 0.01, 0.03, 0.1, 0.2, 0.3, 0.5, 1.0, np.inf]
    err = (~c).float()
    tot_err = err.sum().item()
    print(f"{label}: acc {c.float().mean():.4f}")
    for name, v in (("min arrival gap", g), ("min final gap (m1-m2, m2-m3)", fg)):
        cells = []
        for a, b in zip(edges[:-1], edges[1:]):
            s = (v >= a) & (v < b)
            cells.append(f"[{a},{b}): mass {s.float().mean():.3f} acc {c[s].float().mean():.3f} err share {err[s].sum().item()/tot_err:.2f}")
        print(f"  {name}:\n    " + "\n    ".join(cells))
    # inputs with 2nd max < 0
    neg = xs[:, 1] < 0
    print(f"  2nd max < 0: mass {neg.float().mean():.4f}, acc {c[neg].float().mean():.3f}, err share {err[neg].sum().item()/tot_err:.2f}", flush=True)


with th.no_grad():
    run(example(), "example")
    names, acc, models = load_bench(("mult0.01", "mult0.02"))
    for i in [3, 40]:
        run(models[i], f"{names[i]} (true {acc[i]:.3f})")
