"""Mean max-position code (readout-null part of the final state, per max position), shown in its top PCs."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch as th

from core import example, forward

p = example()
dev = "cuda"
w_oh = th.as_tensor(p[2], device=dev, dtype=th.float64)
Q, _ = th.linalg.qr(w_oh.T)
Pn = th.eye(16, device=dev, dtype=th.float64) - Q @ Q.T
B = th.linalg.svd(Pn)[0][:, :6]                     # orthonormal basis of the readout-null space
N = 2 * 10**6
x = th.randn(N, 10, device=dev, dtype=th.float64, generator=th.Generator(dev).manual_seed(28))
fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
for ax, t in zip(axes, [6, 8, 10]):
    _, H = forward(p, x[:, :t], return_hidden=True)
    pre = x[:, :t]
    o = pre.argsort(1, descending=True)
    m1 = pre.norm(dim=1, keepdim=True)
    z = (H[:, t] @ B) / m1                           # null-space code per unit of input norm
    ages = t - 1 - o[:, 0]
    means = th.stack([z[ages == a].median(0).values for a in range(t)])
    zc = means - means.mean(0)
    U_, S_, V_ = th.linalg.svd(zc)
    pc = (zc @ V_[:2].T).cpu().numpy()
    ax.plot(pc[:, 0], pc[:, 1], "-", color="0.7", zorder=1)
    sc = ax.scatter(pc[:, 0], pc[:, 1], c=np.arange(t), cmap="viridis", s=60, zorder=2)
    for a in range(t):
        ax.annotate(str(a), pc[a], textcoords="offset points", xytext=(5, 5), fontsize=9)
    var = (S_**2 / (S_**2).sum()).cpu().numpy()
    ax.set_title(f"after {t} inputs: mean code by age of the max\n(top 2 PCs explain {var[:2].sum():.0%} of the spread)", fontsize=10)
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
fig.suptitle("Max-position code in the 6 readout-null directions, divided by |x| (medians) (labels = steps since the max arrived)")
fig.tight_layout()
fig.savefig("/data/output/research/algzoo/20261016_m16_10/reports/max_clock_pcs.png", dpi=110)
print("saved")
