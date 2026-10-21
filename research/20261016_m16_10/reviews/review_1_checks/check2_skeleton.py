"""Record skeleton test.

Call x_t 'ordinary' if it arrived below the running 2nd max (relative rank >= 3 at arrival).
An ordinary value can never later enter the running top two, so resampling it anywhere below
the running 2nd max at its arrival leaves every running top-2 value and time unchanged (the
'record skeleton': all values that were top two on arrival, and when).
Share of Var(c) from resampling all ordinary values = how much correctness depends on them.
"""
import sys

sys.path.insert(0, "/data/output/research/algzoo/20261016_m16_10/scripts")
import torch as th

from core import example, forward, target
from bench import load_bench

dev = "cuda"
N = 2**21
normal = th.distributions.Normal(0.0, 1.0)


def correct(params, x):
    return (forward(params, x).argmax(-1) == target(x))


def running_m2(x):
    """Running 2nd max just before each step (-inf before two values have arrived)."""
    n = x.shape[1]
    m1 = th.full((x.shape[0],), -th.inf, device=x.device)
    m2 = th.full_like(m1, -th.inf)
    out = []
    for t in range(n):
        out.append(m2.clone())
        xt = x[:, t]
        new_m1 = th.maximum(m1, xt)
        m2 = th.where(xt > m1, m1, th.maximum(m2, xt))
        m1 = new_m1
    return th.stack(out, 1)


def run(params, label, seed=0):
    g = th.Generator(dev).manual_seed(seed)
    x = th.randn(N, 10, device=dev, generator=g)
    c0 = correct(params, x)
    p = c0.float().mean().item()
    v = p * (1 - p)
    r2 = running_m2(x)
    ordinary = x < r2                      # arrived below running 2nd max
    n_ord = ordinary.float().sum(1).mean().item()
    u = th.rand(N, 10, device=dev, generator=g) * normal.cdf(r2.clamp(min=-8))
    xr = normal.icdf(u.clamp(1e-12, 1 - 1e-7))
    x1 = th.where(ordinary, xr, x)
    # sanity: skeleton unchanged
    assert th.equal(running_m2(x1)[~ordinary], r2[~ordinary])
    c1 = correct(params, x1)
    share = (c1 != c0).float().mean().item() / 2 / v
    # errors by whether the final 3rd max arrived as ordinary
    order = x.argsort(-1, descending=True)
    p3 = order[:, 2]
    m3_ord = ordinary.gather(1, p3[:, None])[:, 0]
    err = ~c0
    print(f"{label}: acc {p:.4f}; mean # ordinary values {n_ord:.2f} of 10;"
          f" Var share from resampling all ordinary values {share:.2f};"
          f" 3rd max ordinary-at-arrival in {m3_ord.float().mean():.2f} of inputs,"
          f" {m3_ord[err].float().mean():.2f} of errors;"
          f" acc when 3rd max ordinary {c0[m3_ord].float().mean():.4f}, when record {c0[~m3_ord].float().mean():.4f}",
          flush=True)


with th.no_grad():
    run(example(), "example")
    names, acc, models = load_bench(("mult0.005", "mult0.01", "mult0.02"))
    for i in [0, 35, 72]:
        run(models[i], f"{names[i]} (true {acc[i]:.3f})")
