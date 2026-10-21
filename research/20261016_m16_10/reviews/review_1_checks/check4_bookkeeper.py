"""Does the network answer what an ideal bookkeeper would answer from the clean circuit's perceived events?

Perceived new max at step t: n7 == 0 after reading x_t (x_t beat the stored max), variant B: n4 > 0.
Perceived new 2nd: otherwise n8 == 0 (x_t beat the stored 2nd).
Bookkeeper: new max -> p2 = p1, p1 = t; new 2nd -> p2 = t. Answer p2.
"""
import sys

sys.path.insert(0, "/data/output/research/algzoo/20261016_m16_10/scripts")
import numpy as np
import torch as th

from core import example, forward, target
from bench import load_bench

dev = "cuda"
N = 2**18


def bookkeeper(H, newmax_fn):
    n = H.shape[1] - 1
    B = H.shape[0]
    p1 = th.full((B,), -1, device=dev)
    p2 = th.full((B,), -1, device=dev)
    for t in range(n):
        h = H[:, t + 1]
        nm = newmax_fn(h)
        n2nd = (~nm) & (h[:, 8] == 0)
        tt = th.full_like(p1, t)
        p2 = th.where(nm, p1, th.where(n2nd, tt, p2))
        p1 = th.where(nm, tt, p1)
    return p2


def evaluate(params, x):
    logits, H = forward(params, x, return_hidden=True)
    out = logits.argmax(-1)
    y = target(x)
    bA = bookkeeper(H, lambda h: h[:, 7] == 0)
    bB = bookkeeper(H, lambda h: h[:, 4] > 0)
    return [(out == y).float().mean().item(), (bA == y).float().mean().item(), (bB == y).float().mean().item(),
            (bA == out).float().mean().item(), (bB == out).float().mean().item()]


with th.no_grad():
    x = th.randn(N, 10, device=dev, generator=th.Generator(dev).manual_seed(3))
    r = evaluate(example(), x)
    print("example: net acc %.4f | bookkeeper acc A %.4f B %.4f | agreement with net A %.4f B %.4f" % tuple(r))
    names, acc, models = load_bench(("mult", "zoo"))
    rows = np.array([evaluate(m, x) for m in models])
    groups = sorted(set(str(n).rsplit("_", 1)[0] if str(n).startswith("mult") else "zoo" for n in names))
    for g in groups:
        s = np.array([(str(n).startswith(g)) for n in names])
        rr = rows[s]
        print(f"{g:10s} net acc {rr[:,0].mean():.3f} | bookkeeper A {rr[:,1].mean():.3f} B {rr[:,2].mean():.3f}"
              f" | agreement A {rr[:,3].mean():.3f} (min {rr[:,3].min():.3f}) B {rr[:,4].mean():.3f}"
              f" | RMSE(bookkeeper A acc - net acc) {np.sqrt(np.mean((rr[:,1]-rr[:,0])**2)):.3f}"
              f" corr {np.corrcoef(rr[:,1], rr[:,0])[0,1]:.2f}")
    m = np.array([str(n).startswith("mult") for n in names])
    rr = rows[m]
    print(f"all 96 noisy: RMSE(bookkeeper A - net) {np.sqrt(np.mean((rr[:,1]-rr[:,0])**2)):.3f},"
          f" corr {np.corrcoef(rr[:,1], rr[:,0])[0,1]:.3f}; constant-guess RMSE {rr[:,0].std():.3f}")
