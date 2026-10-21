"""Which lower values move correctness? Resample one non-top-3 value (kept below m3) and count flips.

For a binary outcome c, resampling x_j from its conditional law given the rest gives
P(flip) = 2 E[p(1-p)] where p = P(c | rest), so P(flip)/2 is the variance of c left by knowing all but x_j.
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


def run(params, label, seed=0):
    g = th.Generator(dev).manual_seed(seed)
    x = th.randn(N, 10, device=dev, generator=g)
    c0 = correct(params, x)
    p = c0.float().mean().item()
    xs, order = x.sort(-1, descending=True)
    m3 = xs[:, 2]
    top3 = th.zeros_like(x, dtype=th.bool)
    top3.scatter_(1, order[:, :3], True)
    out = [f"{label}: acc {p:.4f}, Var(c) {p*(1-p):.4f}"]
    row = []
    for j in range(10):
        sel = ~top3[:, j]
        x1 = x.clone()
        u = th.rand(N, device=dev, generator=g) * normal.cdf(m3)
        x1[:, j] = th.where(sel, normal.icdf(u.clamp(1e-12, 1 - 1e-7)), x[:, j])
        c1 = correct(params, x1)
        flip = (c1 != c0)[sel].float().mean().item()
        row.append(flip / 2)
    out.append("  E[Var(c | all but x_j)] for j not in top3, by position j=0..9: "
               + " ".join(f"{v:.4f}" for v in row))
    # unconditional resample of x_9 (any rank): variance left by knowing x_0..x_8
    x1 = x.clone()
    x1[:, 9] = th.randn(N, device=dev, generator=g)
    flip9 = (correct(params, x1) != c0).float().mean().item()
    out.append(f"  E[Var(c | x_0..x_8)] = {flip9/2:.4f}  (share of Var(c): {flip9/2/(p*(1-p)):.2f})")
    # all 7 lower values jointly resampled below m3
    x1 = x.clone()
    u = th.rand(N, 10, device=dev, generator=g) * normal.cdf(m3)[:, None]
    x1 = th.where(top3, x, normal.icdf(u.clamp(1e-12, 1 - 1e-7)))
    flipall = (correct(params, x1) != c0).float().mean().item()
    out.append(f"  E[Var(c | top-3 values+positions)] = {flipall/2:.4f}  (share {flipall/2/(p*(1-p)):.2f})")
    # only the lower values among the last two positions resampled
    x1 = x.clone()
    last2 = th.zeros_like(top3); last2[:, 8:] = True
    x1 = th.where(top3 | ~last2, x, normal.icdf(u.clamp(1e-12, 1 - 1e-7)))
    flip_l2 = (correct(params, x1) != c0).float().mean().item()
    x1 = x.clone()
    early = th.zeros_like(top3); early[:, :8] = True
    x1 = th.where(top3 | ~early, x, normal.icdf(u.clamp(1e-12, 1 - 1e-7)))
    flip_e = (correct(params, x1) != c0).float().mean().item()
    out.append(f"  resample lower values only at positions 8,9: Var share {flip_l2/2/(p*(1-p)):.2f};"
               f" only at positions 0..7: {flip_e/2/(p*(1-p)):.2f}")
    print("\n".join(out), flush=True)


with th.no_grad():
    run(example(), "example")
    names, acc, models = load_bench(("mult0.01", "mult0.02"))
    for i in [3, 40]:
        run(models[i], f"{names[i]} (true acc {acc[i]:.3f})")
