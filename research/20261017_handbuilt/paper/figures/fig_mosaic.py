"""Thumbnail / figure: does the background decide a close call? 12 near-tie lists (third within 5% of the second,
second positive), the first ones drawn with seed 2026, in draw order. For each, the three largest numbers are kept and
the other seven are redrawn 30 times from a standard normal truncated below the third. Each cell is the network's
answer on one redraw: the second (right), the third, or another position."""
import numpy as np, torch as th, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from algzoo.core import example, forward
from algzoo.handbuilt_v23 import build_v23
NETS = {"ARC's network": [np.asarray(p, float) for p in example()],
        "Hand-built network, 16 neurons": [np.asarray(p, float) for p in build_v23(U_mem=1.0, share=0.25, rho=0.99, level_exact=False).params]}
rng = np.random.default_rng(2026); N, R = 12, 30
lists = []
while len(lists) < N:
    x = rng.standard_normal(10); v = np.sort(x)[::-1]
    if v[1] > 0 and v[1] - v[2] < 0.05 * v[1]: lists.append(x)
X, ranks = [], []
for x in lists:
    o = np.argsort(-x); small = o[3:]; third = x[o[2]]
    for _ in range(R):
        y = x.copy()
        for i in small:
            z = rng.standard_normal()
            while z >= third: z = rng.standard_normal()
            y[i] = z
        X.append(y); ranks.append((o[1], o[2]))
X = th.tensor(np.array(X)); ranks = np.array(ranks)
grids = {}
for name, P in NETS.items():
    a = forward(P, X).argmax(-1).numpy()
    code = np.where(a == ranks[:, 0], 0, np.where(a == ranks[:, 1], 1, 2))
    grids[name] = code.reshape(N, R).T          # rows: redraws, columns: lists
    flips = np.mean([np.mean(grids[name][:, j] != np.bincount(grids[name][:, j], minlength=3).argmax()) for j in range(N)])
    print(f"{name}: answers that differ from the column's usual answer {100*flips:.1f}%; names the third {100*np.mean(code==1):.1f}%")
INK, MUTED = "#222b32", "#566572"
cmap = ListedColormap(["#2a78d6", "#eb6834", "#d8dde1"])
# worked example: among lists whose largest is at least 1.1 x the second, the ARC row closest to half flipped (an illustration; the panels show all 12 lists unselected)
ga = grids["ARC's network"]; share = np.array([np.mean(ga[:, j] != np.bincount(ga[:, j], minlength=3).argmax()) for j in range(N)])
clear = np.array([np.sort(x)[-1] / np.sort(x)[-2] >= 1.1 for x in lists])   # largest clearly above the second
ex = int(np.argmin(np.where(clear, np.abs(share - 0.5), 9))); x0 = lists[ex]; o0 = np.argsort(-x0); v1 = X[ex * R].numpy()
fig = plt.figure(figsize=(10, 8.6), facecolor="white")
gs = fig.add_gridspec(3, 1, height_ratios=[1.25, 3, 3], hspace=0.45)
axe = fig.add_subplot(gs[0]); axe.set_xlim(-2.6, 10.5); axe.set_ylim(-0.25, 2.35); axe.axis("off")
role = {o0[0]: ("largest", "#3b4650"), o0[1]: ("second", "#2a78d6"), o0[2]: ("third", "#eb6834")}
for r, (vals, lab) in enumerate([(x0, "a list with a\nclose call"), (v1, "one variant:\nsmaller 7 redrawn")]):
    y = 1.45 - 1.2 * r
    axe.text(-0.3, y + 0.35, lab, ha="right", va="center", fontsize=9.5, color=MUTED)
    for i, v in enumerate(vals):
        kept = i in role; col = role[i][1] if kept else "#ffffff"
        axe.add_patch(plt.Rectangle((i, y), 0.92, 0.7, facecolor=col if kept else "white", edgecolor="#b8c8d1", linewidth=1))
        axe.text(i + 0.46, y + 0.35, f"{v:.2f}", ha="center", va="center", fontsize=9, color="white" if kept else (INK if r == 0 else "#8a959e"),
                 style="normal" if (kept or r == 0) else "italic")
for i, (name, colr) in role.items():
    axe.text(i + 0.46, 2.27, name, ha="center", va="center", fontsize=8.5, color=colr)
axe.annotate("", xy=(10.35, 0.6), xytext=(9.95, 0.6), arrowprops=dict(arrowstyle="->", color=MUTED))
axe.text(10.45, 0.6, "network's answer\n= one square", ha="left", va="center", fontsize=9, color=MUTED)
axs = [fig.add_subplot(gs[1]), fig.add_subplot(gs[2])]
for ax, (name, g) in zip(axs, grids.items()):
    ax.pcolormesh(g.T, cmap=cmap, vmin=-0.5, vmax=2.5, edgecolors="white", linewidth=1.2)
    ax.set_title(name, fontsize=14, color=INK, loc="left", pad=6); ax.invert_yaxis(); ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([ex + 0.5]); ax.set_yticklabels(["example →"], fontsize=8.5, color=MUTED); ax.tick_params(length=0)
    for sp in ax.spines.values(): sp.set_visible(False)
    ax.set_ylabel("12 different lists", fontsize=10, color=MUTED)
axs[1].set_xlabel("30 variants of each list (three largest kept, the other seven redrawn)", fontsize=10, color=MUTED)
fig.legend(handles=[Patch(color="#2a78d6", label="names the second (right)"), Patch(color="#eb6834", label="names the third"),
                    Patch(color="#d8dde1", label="names another position")], loc="lower center", ncol=3, frameon=False, fontsize=10, labelcolor=INK)
fig.suptitle("Change only the seven smaller numbers: does the network's answer change?", fontsize=13, color=INK, y=0.985)
fig.subplots_adjust(left=0.13, right=0.86, top=0.93, bottom=0.08)
fig.savefig("fig_mosaic.png", dpi=200, facecolor="white"); fig.savefig("fig_mosaic.svg", facecolor="white")
print("saved")
