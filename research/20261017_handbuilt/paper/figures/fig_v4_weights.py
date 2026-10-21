"""Figure 2 of post v4: ARC's 432 weights beside the hand-built network's, neurons grouped and coloured by job, plain labels.
The hand-built network (twin16, 93%) is relabelled and rescaled to line up with ARC (exact symmetries, same function)."""
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Patch
sys.path.insert(0, "/data/output/src/algzoo/src")
from algzoo.core import example
ORDER = [2, 4, 5, 9, 1, 6, 7, 8, 0, 10, 11, 12, 13, 14, 15, 3]
COL = {"store": "#b7791f", "gate": "#0f8a7e", "hiding": "#c2416b", "memory": "#2f6fd0"}
GROUP = {2: "store", 4: "store", 5: "gate", 9: "gate", 1: "hiding", 6: "hiding", 7: "hiding", 8: "hiding",
         **{k: "memory" for k in (0, 10, 11, 12, 13, 14, 15, 3)}}
ARC_LAB = {2: "largest so far", 4: "new largest", 5: "keep gate", 9: "late gate", 1: "hiding", 6: "hiding", 7: "hiding", 8: "hiding",
           **{k: "memory" for k in (0, 10, 11, 12, 13, 14, 15)}, 3: "memory + gap"}
HB_LAB = {**ARC_LAB, 9: "vote shortcut", 3: "memory"}
INK, MUTED = "#18212b", "#5b6876"
T = [np.asarray(p, float) for p in example()]
d = np.load("twin16_matched.npz"); H = [d["wi"], d["wh"], d["wo"]]
def acc(P, n=2_000_000):
    rng = np.random.default_rng(0); x = rng.standard_normal((n, 10)); h = np.zeros((n, 16))
    for t in range(10): h = np.maximum(0, h @ P[1].T + x[:, t:t + 1] * P[0])
    return ((h @ P[2].T).argmax(1) == x.argsort(1)[:, -2]).mean()
LIM = [13, 1.8, 7]   # colour limits: input, recurrent, readout (values beyond are drawn at the limit)
fig = plt.figure(figsize=(15, 13.5), facecolor="white")
gs = fig.add_gridspec(2, 3, width_ratios=[1.2, 16, 10], wspace=0.06, hspace=0.32, left=0.13, right=0.97, top=0.9, bottom=0.1)
for r, (P, title, lab) in enumerate([(T, "Trained by ARC", ARC_LAB), (H, "Built by hand from the explanation", HB_LAB)]):
    wi, wh, wo = P[0][ORDER], P[1][np.ix_(ORDER, ORDER)], P[2][:, ORDER]
    blocks = [(wi[:, None], "from the\ninput", None), (wh, "from neuron …", [str(k) for k in ORDER]), (wo.T, "to the score of position …", [str(k + 1) for k in range(10)])]
    for c, (W, xl, xt) in enumerate(blocks):
        ax = fig.add_subplot(gs[r, c])
        im = ax.imshow(np.clip(W, -LIM[c], LIM[c]), cmap="RdBu_r", vmin=-LIM[c], vmax=LIM[c], aspect="auto", interpolation="nearest")
        ax.set_xticks(range(W.shape[1])); ax.set_xticklabels(xt if xt else [""], fontsize=9, color=INK)
        ax.set_yticks(range(16)); ax.set_yticklabels([f"{lab[k]}  ({k})" for k in ORDER] if c == 0 else [], fontsize=9.5, color=INK)
        ax.tick_params(length=0); ax.xaxis.set_label_position("top"); ax.set_xlabel(xl, fontsize=10, color=MUTED, labelpad=22 if c == 1 else 6)
        for b in (2, 4, 8): ax.axhline(b - 0.5, color="white", lw=2.5)
        if c == 1:
            for b in (2, 4, 8): ax.axvline(b - 0.5, color="white", lw=2.5)
            for j, k in enumerate(ORDER): ax.add_patch(Rectangle((j - 0.5, -1.15), 1, 0.45, color=COL[GROUP[k]], clip_on=False))
        if c == 0:
            for i, k in enumerate(ORDER):
                ax.get_yticklabels()[i].set_color(COL[GROUP[k]]); ax.get_yticklabels()[i].set_fontweight("bold")
        for s in ax.spines.values(): s.set_visible(False)
        if r == 1:
            cb = fig.colorbar(im, ax=ax, orientation="horizontal", fraction=0.035, pad=0.06, aspect=30)
            cb.ax.tick_params(labelsize=8, colors=MUTED); cb.outline.set_visible(False)
    fig.text(0.55, 0.935 if r == 0 else 0.485, f"{title}: right on {100*acc(P):.1f}% of lists", ha="center", fontsize=14, weight="bold", color=INK)
fig.legend(handles=[Patch(color=COL[g], label=n) for g, n in [("store", "store: the largest so far"), ("gate", "gates"), ("hiding", "hiding neurons"), ("memory", "memory")]],
           loc="lower center", ncol=4, frameon=False, fontsize=11, bbox_to_anchor=(0.55, 0.005))
fig.text(0.55, 0.045, "Each row is one neuron and the weights coming into it; the last block is how strongly that neuron adds to each position's score.\n"
         "Red: positive, blue: negative. Each block has its own colour scale (bars below); larger values are drawn at the end of the scale.",
         ha="center", fontsize=10, color=MUTED)
fig.savefig("fig_v4_weights.png", dpi=170, facecolor="white"); print("saved")
