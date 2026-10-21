"""Per (pos1, pos2) linear fits of the final hidden state on the top-3 values; heatmaps of coefficients."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch as th

from core import example, forward

p = example()
dev = "cuda"
s = np.load("/data/output/research/algzoo/20261016_m16_10/reports/scale_mixed.npy")
N = 4 * 10**6
x = th.randn(N, 10, device=dev, dtype=th.float64, generator=th.Generator(dev).manual_seed(14))
_, H = forward(p, x, return_hidden=True)
h = H[:, 10] / th.as_tensor(s, device=dev)
order = x.argsort(-1, descending=True)
xs = x.gather(1, order)
pos1, pos2 = order[:, 0], order[:, 1]
F = th.stack([xs[:, 0], xs[:, 1], xs[:, 2]], 1)
coef = np.full((10, 10, 3, 16), np.nan)
r2 = np.full((10, 10, 16), np.nan)
for i in range(10):
    for j in range(10):
        if i == j:
            continue
        sel = (pos1 == i) & (pos2 == j)
        sol = th.linalg.lstsq(F[sel], h[sel]).solution
        coef[i, j] = sol.cpu().numpy()
        res = h[sel] - F[sel] @ sol
        r2[i, j] = (1 - res.var(0) / h[sel].var(0).clamp(min=1e-12)).cpu().numpy()
np.savez("/data/output/research/algzoo/20261016_m16_10/reports/codes_t10.npz", coef=coef, r2=r2)
order_n = [2, 4, 7, 6, 1, 8, 5, 9, 0, 3, 10, 11, 12, 13, 14, 15]
for k, name in enumerate(["m1", "m2"]):
    fig, axes = plt.subplots(2, 8, figsize=(20, 5.6))
    for ax, n in zip(axes.flat, order_n):
        c = coef[:, :, k, n]
        v = np.nanmax(np.abs(c))
        ax.imshow(c, cmap="RdBu_r", vmin=-v, vmax=v)
        ax.set_title(f"n{n}  (max |c| {v:.2f})", fontsize=9)
        ax.set_xticks(range(10)); ax.set_yticks(range(10))
        ax.tick_params(labelsize=6)
        ax.set_xlabel("pos of 2nd max", fontsize=7); ax.set_ylabel("pos of max", fontsize=7)
    fig.suptitle(f"Final activation (neuron scaled as in the notes): coefficient on {name}, per (pos of max, pos of 2nd max)")
    fig.tight_layout()
    fig.savefig(f"/data/output/research/algzoo/20261016_m16_10/reports/code_coef_{name}.png", dpi=110)
print("mean R2 per neuron:", np.round(np.nanmean(r2, axis=(0, 1)), 3))
