"""Dense version of fig_mosaic: 400 random near-tie lists (third within 5% of the second, second positive; seed 11),
each with 40 variants whose seven smaller numbers are redrawn below the third. One pixel per run, coloured by the
answer. Two versions: lists in draw order, and lists sorted by how often their answer differs from the list's usual
answer (each panel sorted by its own network)."""
import numpy as np, torch as th, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from algzoo.core import example, forward
from algzoo.handbuilt_v23 import build_v23
NETS = {"ARC's network": [np.asarray(p, float) for p in example()],
        "Hand-built network, 16 neurons": [np.asarray(p, float) for p in build_v23(U_mem=1.0, share=0.25, rho=0.99, level_exact=False).params]}
rng = np.random.default_rng(11); N, R = 400, 40; L = []
while len(L) < N:
    x = rng.standard_normal(10); v = np.sort(x)[::-1]
    if v[1] > 0 and v[1] - v[2] < 0.05 * v[1]: L.append(x)
X, rk = [], []
for x in L:
    o = np.argsort(-x); third = x[o[2]]
    for _ in range(R):
        y = x.copy(); z = rng.standard_normal(7)
        while (bad := z >= third).any(): z[bad] = rng.standard_normal(bad.sum())
        y[o[3:]] = z; X.append(y); rk.append((o[1], o[2]))
X = th.tensor(np.array(X)); rk = np.array(rk)
G = {}
for n, P in NETS.items():
    a = forward(P, X).argmax(-1).numpy()
    G[n] = np.where(a == rk[:, 0], 0, np.where(a == rk[:, 1], 1, 2)).reshape(N, R)
    mode = np.array([np.bincount(r, minlength=3).argmax() for r in G[n]])
    print(f"{n}: pixels differing from their list's usual answer {100*(G[n] != mode[:, None]).mean():.1f}%; right {100*(G[n]==0).mean():.1f}%")
INK, MUTED = "#222b32", "#566572"; cmap = ListedColormap(["#2a78d6", "#eb6834", "#d8dde1"])
def order(g):
    mode = np.array([np.bincount(r, minlength=3).argmax() for r in g]); dev = (g != mode[:, None]).mean(1)
    return np.lexsort((mode, dev))[::-1]
for tag, sort in (("", False), ("_sorted", True)):
    fig, axs = plt.subplots(2, 1, figsize=(10, 4.2), facecolor="white")
    for ax, (n, g) in zip(axs, G.items()):
        gg = g[order(g)] if sort else g
        ax.imshow(gg.T, cmap=cmap, vmin=-0.5, vmax=2.5, aspect="auto", interpolation="nearest")
        ax.set_title(n, fontsize=12, color=INK, loc="left", pad=4); ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values(): sp.set_visible(False)
        ax.set_ylabel("40 variants", fontsize=9, color=MUTED)
    axs[1].set_xlabel(f"400 random lists with a close call{', sorted by how often the answer changes' if sort else ''}", fontsize=9.5, color=MUTED)
    fig.legend(handles=[Patch(color="#2a78d6", label="names the second (right)"), Patch(color="#eb6834", label="names the third"),
                        Patch(color="#d8dde1", label="names another position")], loc="lower center", ncol=3, frameon=False, fontsize=9.5, labelcolor=INK)
    fig.suptitle("Each column: one list. Each pixel: the list with its seven smaller numbers redrawn.", fontsize=11.5, color=INK, y=0.99)
    fig.tight_layout(rect=(0, 0.07, 1, 0.95)); fig.savefig(f"fig_mosaic_dense{tag}.png", dpi=200, facecolor="white")
print("saved")
