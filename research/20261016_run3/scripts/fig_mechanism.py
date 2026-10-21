"""Figure: how ARC's model keeps track of position (run 3)."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm  # noqa: E402

from algzoo.core import example  # noqa: E402

BLUE, ORANGE, INK, INK2, GRID, SURF = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "text.color": INK, "font.size": 9,
                     "axes.spines.top": False, "axes.spines.right": False})
w_hi, w_hh, w_oh = (np.asarray(p, np.float64) for p in example())
C = [1, 2, 4, 6, 7, 8]
Dn = [i for i in range(16) if i not in C]
Wc = w_oh - w_oh.mean(0)
h = np.maximum(0, w_hi)
H = [h]
for k in range(1, 12):
    h = np.maximum(0, w_hh @ h)
    H.append(h)
H = np.array(H)

fig, axes = plt.subplots(1, 4, figsize=(17.5, 4.3), constrained_layout=True, gridspec_kw={"width_ratios": [1.15, 1, 1.1, 1.1]})
# A: clock
order = [8, 15, 12, 3, 11, 7, 10, 5, 2, 13, 6, 1, 14, 0, 9, 4]
ax = axes[0]
seq = LinearSegmentedColormap.from_list("seq", [SURF, "#9ec5f0", BLUE, "#123a6b"])
im = ax.imshow(H[:, order].T, aspect="auto", cmap=seq, interpolation="nearest")
ax.set_yticks(range(16), [f"n{i}" + (" (compare)" if i in C else "") for i in order], fontsize=7.5)
ax.set_xticks(range(12))
ax.set_xlabel("steps since a single input of +1")
ax.set_title("A lone input starts a clock:\neach neuron fires in its own part of the cycle", loc="left", fontsize=10)
cb = fig.colorbar(im, ax=ax, shrink=0.8)
cb.set_label("neuron value", color=INK2)
cb.outline.set_visible(False)
# B: two copies cancel
ax = axes[1]
ages = np.arange(1, 10)
comp = np.array([np.linalg.norm(Wc[:, C] @ H[a, C]) for a in ages])
oth = np.array([np.linalg.norm(Wc[:, Dn] @ H[a, Dn]) for a in ages])
tot = np.array([np.linalg.norm(Wc @ H[a]) for a in ages])
ax.plot(ages, comp, color=BLUE, lw=2, marker="o", ms=5, mec=SURF, label="what the output sees of the comparing neurons")
ax.plot(ages, oth, color=ORANGE, lw=2, marker="o", ms=5, mec=SURF, label="what it sees of the other ten neurons")
ax.plot(ages, tot, color=INK2, lw=2, marker="o", ms=5, mec=SURF, label="what it sees in total")
ax.set_ylim(0, 180)
ax.set_xlabel("steps since a single input")
ax.set_ylabel("size of the score differences")
ax.grid(True, color=GRID, lw=0.8)
ax.set_title("The clock is invisible to the output:\nits two parts cancel, so a lone input casts no vote", loc="left", fontsize=10)
ax.legend(frameon=False, fontsize=7.5, loc="center", bbox_to_anchor=(0.5, 0.3))
# C: sliding vote
ax = axes[2]
x = np.zeros(10)
x[2] = 1.0
x[5] = 0.7
hh = np.zeros(16)
S = []
for t in range(10):
    hh = np.maximum(0, w_hh @ hh + x[t] * w_hi)
    s = w_oh @ hh
    S.append(s - s.mean())
S = np.array(S)[2:]
div = LinearSegmentedColormap.from_list("div", [BLUE, "#cfd8e6", "#ece9e4", "#f3cdb8", ORANGE])
im = ax.imshow(S, aspect="auto", cmap=div, norm=TwoSlopeNorm(0, -60, 60), interpolation="nearest")
ax.set_yticks(range(8), [f"after input {t}" for t in range(2, 10)], fontsize=7.5)
ax.set_xticks(range(10))
ax.set_xlabel("position the score votes for")
for t in range(5, 10):
    ax.add_patch(plt.Rectangle((5 + 9 - t - 0.5, t - 2 - 0.5), 1, 1, fill=False, ec=INK, lw=1.2))
ax.set_title("Votes slide one position per step:\nlargest at 2, second (0.7) at 5", loc="left", fontsize=10)
cb = fig.colorbar(im, ax=ax, shrink=0.8)
cb.set_label("score minus average", color=INK2)
cb.outline.set_visible(False)
# D: second first; the largest's arrival reveals the vote
ax = axes[3]
x = np.zeros(10)
x[2] = 0.7
x[6] = 1.0
hh = np.zeros(16)
S = []
for t in range(10):
    hh = np.maximum(0, w_hh @ hh + x[t] * w_hi)
    s = w_oh @ hh
    S.append(s - s.mean())
S = np.array(S)[2:]
im = ax.imshow(S, aspect="auto", cmap=div, norm=TwoSlopeNorm(0, -60, 60), interpolation="nearest")
ax.set_yticks(range(8), [f"after input {t}" for t in range(2, 10)], fontsize=7.5)
ax.set_xticks(range(10))
ax.set_xlabel("position the score votes for")
for t in range(6, 10):
    ax.add_patch(plt.Rectangle((2 + 9 - t - 0.5, t - 2 - 0.5), 1, 1, fill=False, ec=INK, lw=1.2))
ax.set_title("The largest arriving later reveals the vote:\nsecond (0.7) at 2, largest at 6", loc="left", fontsize=10)
cb = fig.colorbar(im, ax=ax, shrink=0.8)
cb.set_label("score minus average", color=INK2)
cb.outline.set_visible(False)
fig.savefig("/data/output/research/algzoo/20261016_run3/reports/fig_mechanism.png", dpi=150)
print("saved")
