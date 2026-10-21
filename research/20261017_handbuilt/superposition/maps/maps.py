"""Decision maps; see README.md."""
import numpy as np, torch as th, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from scipy.ndimage import label
from algzoo.core import example, forward
from algzoo.handbuilt_v23 import build_v23
NETS = {"ARC's network": [np.asarray(p, float) for p in example()],
        "Hand-built, 16 neurons": [np.asarray(p, float) for p in build_v23(U_mem=1.0, share=0.25, rho=0.99, level_exact=False).params]}
rng = np.random.default_rng(11); L = []
while len(L) < 4:
    x = rng.standard_normal(10); v = np.sort(x)[::-1]
    if v[1] > 0 and v[1] - v[2] < 0.05 * v[1]: L.append(x)
K = 241
INK, MUTED = "#222b32", "#566572"; cmap = ListedColormap(["#2a78d6", "#eb6834", "#d8dde1"])
fig, axs = plt.subplots(2, 4, figsize=(13, 7), facecolor="white")
for j, x in enumerate(L):
    o = np.argsort(-x); third = x[o[2]]; small = o[3:]; pa, pb = small[0], small[1]   # the two largest smaller numbers
    g = np.linspace(-3, third - 1e-6, K); A, B = np.meshgrid(g, g)
    X = np.repeat(x[None], K * K, 0); X[:, pa] = A.ravel(); X[:, pb] = B.ravel(); X = th.tensor(X)
    for i, (n, P) in enumerate(NETS.items()):
        a = forward(P, X).argmax(-1).numpy()
        m = np.where(a == o[1], 0, np.where(a == o[2], 1, 2)).reshape(K, K)
        regions = sum(label(m == c)[1] for c in range(3)); share = np.bincount(m.ravel(), minlength=3) / m.size
        sw = (np.diff(m, axis=1) != 0).sum(1).mean()
        print(f"list {j} (largest at {o[0]}, second at {o[1]}, third at {o[2]}, gap {100*(x[o[1]]-third)/x[o[1]]:.1f}%; swept positions {pa}, {pb}) "
              f"{n}: right {100*share[0]:.0f}%, third {100*share[1]:.0f}%, other {100*share[2]:.0f}%; regions {regions}; switches per row {sw:.2f}")
        ax = axs[i, j]; ax.imshow(m, cmap=cmap, vmin=-0.5, vmax=2.5, origin="lower", extent=(g[0], g[-1], g[0], g[-1]), interpolation="nearest")
        ax.set_xticks([-3, 0]); ax.set_yticks([-3, 0]); ax.tick_params(labelsize=8, colors=MUTED, length=2)
        for sp in ax.spines.values(): sp.set_visible(False)
        if i == 0: ax.set_title(f"list {j + 1}: positions {pa} and {pb} swept", fontsize=10, color=INK)
        if j == 0: ax.set_ylabel(n, fontsize=11, color=INK)
fig.legend(handles=[Patch(color="#2a78d6", label="names the second (right)"), Patch(color="#eb6834", label="names the third"),
                    Patch(color="#d8dde1", label="names another position")], loc="lower center", ncol=3, frameon=False, fontsize=10, labelcolor=INK)
fig.suptitle("Four close calls: the answer as two of the smaller numbers vary (axes: their values, from -3 to just below the third)", fontsize=11.5, color=INK)
fig.tight_layout(rect=(0, 0.05, 1, 0.95)); fig.savefig("maps.png", dpi=170, facecolor="white"); print("saved")
