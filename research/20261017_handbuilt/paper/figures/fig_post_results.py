"""Figures for lesswrong_post_results.md, drawn from numbers in the logs named beside each (no recomputation here).
fig_modes: new mistakes by distance from the second when one rotation of the memory is damped (superposition/steer/modes.log).
fig_zone: how often the third wins by its gap to the second (superposition/width_band/curve.log, deduce_width.log).
fig_density: linear pieces and unexplained near-tie decisions across networks (siblings/siblings.log, noise_control/run.log,
finetune/simpler.log, train_from_handbuilt/run.log; twin16 from noise_control/run_twin16.log).
fig_mistakes: share of ARC's mistakes reproduced by references and explanations (superposition/README.md, generality section)."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
# modes
d = np.arange(1, 10)
modes = {"180° (fastest)": [54, 3, 9, 4, 10, 6, 9, 3, 4], "126°": [44, 11, 3, 11, 4, 6, 12, 3, 6], "83°": [20, 15, 10, 10, 7, 12, 12, 7, 7], "43° (slowest)": [11, 9, 10, 8, 10, 19, 17, 9, 6]}
fig, ax = plt.subplots(figsize=(7, 3.8))
for k, (lab, v) in enumerate(modes.items()): ax.plot(d, v, marker="o", label=lab)
ax.set_xlabel("distance from the second-largest to the wrongly named position"); ax.set_ylabel("% of the new mistakes"); ax.legend(title="rotation damped by 20%", frameon=False)
ax.set_title("Damping one rotation of the memory: fast ones confuse neighbours, slow ones far positions", fontsize=10)
fig.tight_layout(); fig.savefig("fig_modes.svg"); fig.savefig("fig_modes.png", dpi=170)
# zone
mids = np.array([0.25, 0.75, 1.5, 2.5, 4, 6.5, 10, 16]); obs = [44.2, 35.2, 25.8, 17.4, 10.6, 5.5, 2.8, 1.3]; pred = [46.3, 37.3, 28.4, 20.9, 13.9, 8.2, 5.1, 3.4]
fig, ax = plt.subplots(figsize=(6.5, 3.8))
ax.plot(mids, obs, "o-", label="the network"); ax.plot(mids, pred, "s--", label="predicted from the weights\n(other numbers as noise)")
ax.set_xlabel("gap between second and third, % of the second"); ax.set_ylabel("% of lists where the third wins"); ax.legend(frameon=False); ax.set_xscale("log")
ax.set_title("The uncertain zone near a tie", fontsize=10); fig.tight_layout(); fig.savefig("fig_zone.svg"); fig.savefig("fig_zone.png", dpi=170)
# density: (maps covering half, tree accuracy, always-guess)
sib = np.load("../../superposition/siblings/siblings.npy")      # acc, third share, maps, gain (rows in siblings.log order)
names = open("../../superposition/siblings/siblings.log").read().splitlines()
pts = {"trained 16 neurons": [], "trained 20 and 32 neurons": [], "ARC": [], "hand-built": []}
for line, row in zip(names, sib):
    acc, third, maps, gain = row; unexpl = (1 - gain) * 100
    if line.startswith("hand-built v19"): continue          # the post's 16-neuron hand-built network is now twin16 (below)
    key = "ARC" if line.startswith("ARC") else "hand-built" if line.startswith("hand") else "trained 20 and 32 neurons" if line.startswith(("20 ", "32", "zoo32")) else "trained 16 neurons"
    pts[key].append((maps, unexpl))
pts["hand-built"].append((87, 100 * (1 - (0.991 - 0.912) / (1 - 0.912))))   # twin16: superposition/noise_control/run_twin16.log (same lists and tree)
extra = {"hand-built + weight noise": [(56, 17), (54, 17), (75, 14), (80, 18), (241, 13), (238, 19), (372, 14), (410, 19)],
         "ARC trained further": [(5532, 11.7)], "hand-built (28) trained further": [(301, None), (218, None)]}
fig, ax = plt.subplots(figsize=(8.5, 4.4))
st = {"trained 16 neurons": ("o", "#888"), "trained 20 and 32 neurons": ("^", "#555"), "ARC": ("*", "k"), "hand-built": ("D", "C2"), "hand-built + weight noise": ("d", "C2")}
for k, v in pts.items():
    v = np.array(v); ax.scatter(v[:, 0], v[:, 1], marker=st[k][0], c=st[k][1], s=140 if k == "ARC" else 30, label=k, zorder=3)
v = np.array(extra["hand-built + weight noise"]); ax.scatter(v[:, 0], v[:, 1], marker="d", facecolors="none", edgecolors="C2", s=40, label="hand-built (28) + weight noise", zorder=3)
ax.scatter([5532], [73], marker="*", facecolors="none", edgecolors="k", s=140, label="ARC trained further", zorder=3)
ax.set_xscale("log"); ax.set_xlabel("linear pieces needed to cover half of all lists (decision precision 3%)")
ax.set_ylabel("% of the near-tie decisions' predictable part\nthat a rule on the three largest misses"); ax.legend(frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1))
ax.set_title("Close calls: trained networks are dense, the hand-built ones stay simple", fontsize=10)
fig.tight_layout(); fig.savefig("fig_density.svg"); fig.savefig("fig_density.png", dpi=170)
# mistakes reproduced
labs = ["an exact solver\nwith small errors", "other trained\nnetworks", "hand-built\nnetworks", "ARC with its smaller\nnumbers redrawn", "ARC with tiny\nweight noise"]
lo = [19, 29, 6, 55, 65]; hi = [24, 35, 17, 65, 79]   # hand-built: hilton/topdown/mistakes_twin16.log (twin16 17%, v21 6%)
fig, ax = plt.subplots(figsize=(8.2, 3.6))
ax.bar(range(5), [(a + b) / 2 for a, b in zip(lo, hi)], yerr=[[(b - a) / 2 for a, b in zip(lo, hi)]] * 2, color=["#bbb", "#bbb", "C0", "#666", "#666"], capsize=4)
ax.set_xticks(range(5)); ax.set_xticklabels(labs, fontsize=8.5); ax.set_ylabel("% of ARC's mistakes repeated")
ax.set_title("Who makes ARC's particular mistakes? (ranges across five trained networks or variants)", fontsize=9.5)
fig.tight_layout(); fig.savefig("fig_mistakes.svg"); fig.savefig("fig_mistakes.png", dpi=170)
print("done")
