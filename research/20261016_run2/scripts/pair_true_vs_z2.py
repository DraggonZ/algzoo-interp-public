"""Diagnostic: per position pair (p1, p2), sampled true accuracy against exact two-spike accuracy Z_2."""
import numpy as np
import torch as th

from algzoo.bench import load_bench
from algzoo.core import example, forward
from algzoo.exact_z2 import exact_z2

dev = "cuda"
names, acc, models = load_bench(("mult",))
get = lambda n: example() if n == "example" else models[list(names).index(n)]
np.set_printoptions(linewidth=220, precision=2, suppress=True)


@th.no_grad()
def pair_acc(params, N=2**24, seed=0):
    x = th.randn(N, 10, device=dev, generator=th.Generator(dev).manual_seed(seed))
    idx = x.topk(2, -1).indices
    c = (forward(params, x).argmax(-1) == idx[:, 1]).float()
    key = idx[:, 0] * 10 + idx[:, 1]
    s = th.bincount(key, weights=c, minlength=100)
    n = th.bincount(key, minlength=100).clamp(min=1)
    return (s / n).reshape(10, 10).cpu().numpy()


allz, allt = [], []
for nm in ["example", "mult0.02_18", "mult0.02_3", "mult0.01_14", "mult0.02_8", "mult0.005_6"]:
    params = get(nm)
    _, _, _, det = exact_z2(params, return_detail=True)
    Z = np.full((10, 10), np.nan)
    for (p1, p2), (a, _) in det.items():
        Z[p1, p2] = a
    T = pair_acc(params)
    np.fill_diagonal(T, np.nan)
    m = ~np.isnan(Z)
    print(f"\n{nm}: Z2 {np.nanmean(Z):.3f} true {np.nanmean(T):.3f}; corr over pairs {np.corrcoef(Z[m], T[m])[0,1]:.3f}")
    print("true minus Z2 per pair (rows p1, cols p2):")
    print(T - Z)
    # how much of the loss is in pairs where Z2 is already low
    lo = Z < 0.5
    print(f"pairs with Z2<0.5: {lo.sum()}; their share of (1-true) {((1 - T)[lo].sum() / (1 - T)[m].sum()):.2f}; mean true there {np.nanmean(T[lo]) if lo.any() else float('nan'):.2f}")
