"""Figure: error against compute for sampling, the keep-the-top-k ladder at its ideal, and the sampling-free estimate.
Numbers from topk_scaling.log (reduction RMSE over 22 trained models), step 87's prefix-tree costs, new_score.log and
reduce3_det.log / shared_vs_sampling.log. Sampling curve uses the mean p(1 - p) of the 22 trained models."""
import glob

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

BLUE, ORANGE, AQUA, INK, INK2, GRID, SURF = "#2a78d6", "#eb6834", "#1baf7a", "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "text.color": INK, "font.size": 9})
acc6 = [0.9532, 0.9101, 0.8804, 0.9485, 0.9320, 0.8898]
new = np.load("reports/new_truth.npy")[:, 1]
pv = np.mean([a * (1 - a) for a in list(acc6) + list(new)])
n = np.logspace(2, 5.4, 200)
fig, ax = plt.subplots(figsize=(6.4, 4.0))
ax.plot(n, 100 * np.sqrt(pv / n), color=BLUE, lw=2, label="sampling")
cost = np.array([306, 1910, 10602, 0.33 * 151200]); err = np.array([0.54, 0.42, 0.35, 0.26])
ax.plot(cost, err, color=ORANGE, lw=2, ls="--", marker="o", ms=8, mec=SURF, mew=2,
        label="keep the top k exactly (ideal: one run per placement, no other error)")
for c, e, k in zip(cost, err, [3, 4, 5, 6]):
    ax.annotate(f"k = {k}", (c, e), textcoords="offset points", xytext=(0, 9), ha="center", color=INK2)
ax.plot([1760], [0.94], marker="o", ms=9, color=AQUA, mec=SURF, mew=2, ls="none", label="sampling-free estimate, 16 new models")
ax.plot([1700], [0.78], marker="o", ms=9, mfc=SURF, mec=AQUA, mew=2, ls="none", label="same estimate, six models it was tuned on")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("compute, in model runs' worth"); ax.set_ylabel("error in accuracy, percentage points (root-mean-square)")
ax.set_yticks([0.1, 0.2, 0.5, 1, 2]); ax.set_yticklabels(["0.1", "0.2", "0.5", "1", "2"])
ax.grid(True, which="major", color=GRID, lw=0.8); ax.set_axisbelow(True)
for sp in ("top", "right"):
    ax.spines[sp].set_visible(False)
ax.legend(frameon=False, loc="lower left", fontsize=8)
ax.set_title("Sampling keeps improving with compute. Keeping the top inputs exactly stalls.", fontsize=10, color=INK, loc="left")
fig.tight_layout()
fig.savefig("reports/fig_challenge.png", dpi=160)
print("saved")
