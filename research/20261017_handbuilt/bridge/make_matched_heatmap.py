"""Hilton-style heatmaps of the trained network (top) and the hand-built network after relabelling and rescaling
(bottom; match_order_scale.py). Same colour limits per matrix in both rows."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from algzoo.core import example

plt.rcParams.update({"mathtext.fontset": "cm"})
T = [np.asarray(p, float) for p in example()]
import sys
d = np.load(sys.argv[1]); M = (d["wi"], d["wh"], d["wo"])

def fmt(v):
    v = round(float(v), 6); a = abs(v)
    if a < 0.005: return "0.00" if a == 0 else f"{v:.0e}".replace("e-0", "e-")
    if a >= 999.5: return ("-" if v < 0 else "") + f"{a:.0e}".replace("e+0", "e")
    if a >= 99.5: return f"{v:.0f}"
    return f"{v:.2f}"

lims = [13.2, 1.75, 7.4]
fig, axs = plt.subplots(2, 3, figsize=(24, 17), dpi=180, gridspec_kw={"width_ratios": [1.5, 9, 6.2], "wspace": 0.25, "hspace": 0.22})
for r, (P, name) in enumerate([(T, "Trained by ARC (95.3%)"), (M, sys.argv[3])]):
    for k, (W, ttl, xl, yl) in enumerate([(P[0][:, None], r"$W^{hi}$", "input neuron", "output neuron"),
                                         (P[1], r"$W^{hh}$", "input neuron", "output neuron"),
                                         (P[2].T, r"$(W^{oh})^{\top}$", "output neuron", "input neuron")]):
        ax = axs[r, k]
        sns.heatmap(np.clip(W, -lims[k], lims[k]), ax=ax, cmap="icefire", center=0, vmin=-lims[k], vmax=lims[k],
                    annot=np.vectorize(fmt)(W), fmt="", annot_kws={"fontsize": 7.5}, cbar_kws={"pad": 0.03})
        ax.set_title(ttl, fontsize=20); ax.set_xlabel(xl, fontsize=11); ax.set_ylabel(yl, fontsize=11)
        ax.set_yticklabels(range(16), rotation=0)
    axs[r, 1].text(8, -1.6, name, ha="center", fontsize=17, weight="bold")
fig.savefig(sys.argv[2], bbox_inches="tight", facecolor="white")
