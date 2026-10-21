"""Run-2 figures: per-pair two-spike error (base vs degraded copy), write vs carry losses, weight-group attribution."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

from algzoo.bench import load_bench  # noqa: E402
from algzoo.core import example  # noqa: E402
from algzoo.exact_z2 import exact_z2  # noqa: E402

OUT = "/data/output/research/algzoo/20261016_run2/reports/"
BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2, "text.color": INK, "font.size": 10,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
})
names, acc, models = load_bench(("mult",))

# 1. per-pair two-spike error, exact
cmap = LinearSegmentedColormap.from_list("blue", ["#fcfcfb"] + RAMP)
fig, axes = plt.subplots(1, 2, figsize=(8.6, 4.6), constrained_layout=True)
for ax, (label, params) in zip(axes, [("base model (Z_2 = 0.977)", example()),
                                      ("noisy copy mult0.02_18 (Z_2 = 0.642)", models[list(names).index("mult0.02_18")])]):
    _, _, _, det = exact_z2(params, return_detail=True)
    M = np.full((10, 10), np.nan)
    for (p1, p2), (a, _) in det.items():
        M[p1, p2] = 1 - a
    ax.grid(False)
    im = ax.imshow(M, cmap=cmap, vmin=0, vmax=1, origin="upper")
    ax.set_xticks(range(10))
    ax.set_yticks(range(10))
    ax.set_xlabel("position of the 2nd largest input")
    ax.set_ylabel("position of the largest input")
    ax.set_title(label, loc="left", fontsize=10)
    for s in ax.spines.values():
        s.set_visible(False)
cb = fig.colorbar(im, ax=axes, shrink=0.8)
cb.set_label("two-spike error (exact)")
cb.outline.set_visible(False)
fig.suptitle("Degraded copies fail whole position pairs, at every ratio of the two values\n(white diagonal: impossible pairs)", x=0.02, ha="left", fontsize=11)
fig.savefig(OUT + "fig_pairs_z2.png", dpi=150)

# 2. write vs carry losses by noise level, from z2_stages
st = np.load(OUT + "z2_stages.npy", allow_pickle=True).item()
sig = np.array([float(str(n).split("_")[0][4:]) for n in names])
write = np.array([1 - st[n][0]["right_T_right_end"] - st[n][0]["right_T_wrong_end"] for n in names])
carry = np.array([st[n][0]["right_T_wrong_end"] for n in names])
fig, ax = plt.subplots(figsize=(6.2, 3.6))
levels = [0.005, 0.01, 0.02]
x = np.arange(3)
for off, vals, col, lab in [(-0.12, write, BLUE, "wrong right after the last spike (write)"),
                            (0.12, carry, ORANGE, "lost afterwards, on zero inputs (carry)")]:
    for i, s in enumerate(levels):
        v = vals[sig == s]
        ax.scatter(np.full(len(v), x[i] + off) + np.random.default_rng(i).uniform(-0.04, 0.04, len(v)), v, s=14, color=col,
                   alpha=0.55, edgecolor="none", zorder=2)
        ax.plot([x[i] + off - 0.08, x[i] + off + 0.08], [v.mean()] * 2, color=col, lw=2.5, zorder=3)
    ax.scatter([], [], color=col, label=lab, s=20)
ax.axhline(1 - st["example"][0]["right_T_right_end"] - st["example"][0]["right_T_wrong_end"], color=BLUE, lw=1, ls=":", zorder=1)
ax.axhline(st["example"][0]["right_T_wrong_end"], color=ORANGE, lw=1, ls=":", zorder=1)
ax.set_xticks(x)
ax.set_xticklabels([f"σ = {s}" for s in levels])
ax.set_ylabel("share of two-spike inputs")
ax.legend(frameon=False, fontsize=9, loc="upper left")
ax.set_title("Noisy copies lose the answer after it is written (dots: copies; bars: means;\ndotted: base model)",
             loc="left", fontsize=10)
fig.tight_layout()
fig.savefig(OUT + "fig_write_carry.png", dpi=150)

# 3. attribution: share of each copy's two-spike loss removed by restoring one weight group
A = np.load(OUT + "z2_attrib.npz")
R, groups = A["R"], list(A["groups"])
zb = 0.977427
loss = zb - R[:, 0, 0]
labels = {"hh_CC": "W_hh among the 6 comparator neurons (36)", "five": "the 5 sensitive weights (5)", "W_oh": "readout (160)",
          "hh_dense_dense": "W_hh among the 10 dense neurons (100)", "hh_dense<-C": "W_hh comparators → dense (60)",
          "hh_C<-dense": "W_hh dense → comparators (60)", "W_hi": "input weights (16)"}
order = ["hh_CC", "five", "W_oh", "hh_dense_dense", "hh_dense<-C", "hh_C<-dense", "W_hi"]
share = [((R[:, groups.index(g), 0] - R[:, 0, 0]).sum() / loss.sum()) for g in order]
fig, ax = plt.subplots(figsize=(7.8, 3.6), constrained_layout=True)
y = np.arange(len(order))[::-1]
ax.barh(y, share, color=BLUE, height=0.6, zorder=2)
for yy, v in zip(y, share):
    ax.annotate(f"{v:.0%}", (v, yy), textcoords="offset points", xytext=(4, -3), color=INK, fontsize=9)
ax.set_yticks(y)
ax.set_yticklabels([labels[g] for g in order])
ax.grid(axis="y", visible=False)
ax.set_xlim(0, 0.85)
ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
ax.set_xlabel("share of the 96 copies' total two-spike loss removed (exact)")
ax.set_title("Restoring one weight group to its base values:\nthe comparator block carries most of the damage", loc="left", fontsize=10)
fig.savefig(OUT + "fig_attrib.png", dpi=150)
print("saved")
