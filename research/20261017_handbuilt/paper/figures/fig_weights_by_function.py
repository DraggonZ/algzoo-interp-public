"""ARC's weights beside a hand-built network's, neurons grouped by function (like the image sent to Hilton, rearranged).
The hand-built network is relabelled and rescaled to line up with ARC (exact symmetries: same function; see
paper/tweet/match_order_scale.py). Usage: python fig_weights_by_function.py [path to a matched .npz with wi, wh, wo]
(default: v18, paper/tweet/handbuilt_matched.npz; the post uses twin16_matched.npz). Rows and columns of W_hh follow the same order."""
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from algzoo.core import example
plt.rcParams.update({"mathtext.fontset": "cm"})
ORDER = [2, 4, 5, 9, 1, 6, 7, 8, 0, 10, 11, 12, 13, 14, 15, 3]
GROUPS = [("largest so far", 2), ("keep gate", 1), ("late gate", 1), ("hiding", 4), ("memory", 8)]
ARC_ROLE = {2: "store", 4: "new largest", 5: "keep gate", 9: "late gate", 1: "hiding", 6: "hiding", 7: "hiding", 8: "hiding + 2nd",
            0: "mem, off at age 2", 10: "mem, off at age 6", 11: "mem, off at age 5", 12: "memory", 13: "memory", 14: "memory",
            15: "mem, content-switched", 3: "mem, content-switched"}
HB_ROLE = {2: "store", 4: "new largest", 5: "keep gate", 9: "flip (hides)", 1: "clock (hides)", 6: "clock (hides)", 7: "clock (hides)",
           8: "flag", **{k: "memory slot" for k in (0, 10, 11, 12, 13, 14, 15, 3)}}
path = sys.argv[1] if len(sys.argv) > 1 else "../tweet/handbuilt_matched.npz"
T = [np.asarray(p, float) for p in example()]; d = np.load(path); M = [d["wi"], d["wh"], d["wo"]]
if "roles" in d: HB_ROLE = {k: str(r) for k, r in enumerate(d["roles"])}   # a matched file may carry its own role names
def acc(P):
    x = torch.randn(4000000, 10, generator=torch.Generator().manual_seed(0), dtype=torch.float64); h = torch.zeros(len(x), 16, dtype=torch.float64)
    wi, wh, wo = (torch.tensor(p) for p in P)
    for t in range(10): h = torch.relu(h @ wh.T + x[:, t:t + 1] * wi)
    return ((h @ wo.T).argmax(1) == x.argsort(1)[:, -2]).double().mean().item()
def fmt(v):
    v = round(float(v), 6); a = abs(v)
    if a == 0: return "0"
    if a < 0.005: return f"{v:.0e}".replace("e-0", "e-")
    if a >= 999.5: return ("-" if v < 0 else "") + f"{a:.0e}".replace("e+0", "e")
    if a >= 99.5: return f"{v:.0f}"
    return f"{v:.2f}"
def reorder(P): return [P[0][ORDER], P[1][np.ix_(ORDER, ORDER)], P[2][:, ORDER]]
lims = [13.2, 1.75, 7.4]
fig, axs = plt.subplots(2, 3, figsize=(25, 18), dpi=170, gridspec_kw={"width_ratios": [1.5, 9, 6.2], "wspace": 0.42, "hspace": 0.25})
bounds = np.cumsum([n for _, n in GROUPS])[:-1]
for r, (P, name, role) in enumerate([(T, f"Trained by ARC ({100*acc(T):.1f}%)", ARC_ROLE), (M, f"Hand-built ({100*acc(M):.1f}%)", HB_ROLE)]):
    Q = reorder(P); labels = [f"{role[k]}  {k}" for k in ORDER]
    for k, (W, ttl, xl) in enumerate([(Q[0][:, None], r"$W^{hi}$  (input)", ""), (Q[1], r"$W^{hh}$  (recurrent: row = receiving neuron)", "sending neuron"), (Q[2].T, r"$(W^{oh})^{\top}$  (readout)", "position scored")]):
        ax = axs[r, k]
        sns.heatmap(np.clip(W, -lims[k], lims[k]), ax=ax, cmap="icefire", center=0, vmin=-lims[k], vmax=lims[k], annot=np.vectorize(fmt)(W), fmt="", annot_kws={"fontsize": 7}, cbar_kws={"pad": 0.02})
        ax.set_title(ttl, fontsize=17); ax.set_xlabel(xl, fontsize=11)
        ax.set_yticks(np.arange(16) + 0.5); ax.set_yticklabels(labels if k == 0 else [str(i) for i in ORDER], rotation=0, fontsize=9)
        for b in bounds:
            ax.axhline(b, color="white", lw=2.2)
            if k == 1: ax.axvline(b, color="white", lw=2.2)
        if k == 1: ax.set_xticks(np.arange(16) + 0.5); ax.set_xticklabels([str(i) for i in ORDER], fontsize=9)
    y = 0
    for gname, nn in GROUPS:
        y += nn
    axs[r, 1].text(8, -1.3, name, ha="center", fontsize=16, weight="bold")
out = path.split("/")[-1].replace(".npz", "")
fig.savefig(f"fig_weights_by_function_{out}.png", bbox_inches="tight", facecolor="white"); fig.savefig(f"fig_weights_by_function_{out}.svg", bbox_inches="tight", facecolor="white")
print("saved", out)
