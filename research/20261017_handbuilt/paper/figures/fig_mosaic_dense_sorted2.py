"""Sorted dense mosaic, second version: same 400 lists and 40 variants as fig_mosaic_dense.py (seed 11). Within each
list the variants are sorted by answer (right on top, then the third, then another position), and lists are sorted by
their share of right answers, then of the third. Each column then shows that list's answer shares. A list whose
answer is decided by the three largest numbers is a single colour; a mixed column means the background matters."""
import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
exec(open("fig_mosaic_dense.py").read().split("INK, MUTED")[0])
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
INK, MUTED = "#222b32", "#566572"; cmap = ListedColormap(["#2a78d6", "#eb6834", "#d8dde1"])
fig, axs = plt.subplots(2, 1, figsize=(10, 4.6), facecolor="white")
for ax, (n, g) in zip(axs, G.items()):
    cnt = np.stack([(g == k).sum(1) for k in range(3)], 1)
    o = np.lexsort((-cnt[:, 1], -cnt[:, 0]))
    cols = np.sort(g[o], axis=1)
    mixed = (cnt.max(1) < 0.9 * R).mean()
    ax.imshow(cols.T, cmap=cmap, vmin=-0.5, vmax=2.5, aspect="auto", interpolation="nearest")
    ax.set_title(f"{n}: the answer depends on the background in {100*mixed:.0f}% of lists", fontsize=11.5, color=INK, loc="left", pad=4)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values(): sp.set_visible(False)
    ax.set_ylabel("40 variants", fontsize=9, color=MUTED)
    print(n, f"mixed {100*mixed:.1f}%  always right {100*(cnt[:,0]==R).mean():.1f}%  always third {100*(cnt[:,1]==R).mean():.1f}%")
axs[1].set_xlabel("400 random lists with a close call, sorted by how often the network gets each one right", fontsize=9.5, color=MUTED)
fig.legend(handles=[Patch(color="#2a78d6", label="names the second (right)"), Patch(color="#eb6834", label="names the third"),
                    Patch(color="#d8dde1", label="names another position")], loc="lower center", ncol=3, frameon=False, fontsize=9.5, labelcolor=INK)
fig.suptitle("Same three largest numbers, the other seven redrawn 40 times: does the answer change?", fontsize=11.5, color=INK, y=0.99)
fig.text(0.5, 0.115, "'Depends on the background': the list's most common answer appears in fewer than 90% of its variants.", ha="center", fontsize=8.5, color=MUTED)
fig.tight_layout(rect=(0, 0.12, 1, 0.95)); fig.savefig("fig_mosaic_dense_sorted2.png", dpi=200, facecolor="white")
print("saved")
