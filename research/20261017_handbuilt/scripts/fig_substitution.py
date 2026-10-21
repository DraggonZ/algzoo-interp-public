"""Figure: variance matched against accuracy with that neuron simulated, every neuron of 22 trained networks and of the
two hand-built ones (role-free formulas, gao/sim_across_points.json), plus v18 simulated from its exact rules."""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
d = json.load(open("gao/sim_across_points.json"))
fig, ax = plt.subplots(figsize=(6.4, 4.2))
for name, v in d.items():
    if name in ("v5", "v18"):
        continue
    xs = [p[1] for p in v["points"]]; ys = [p[2] - v["base"] for p in v["points"]]
    ax.scatter(xs, [100 * y for y in ys], s=9, c="#9ab" if name != "ARC" else "#c33", alpha=0.6 if name != "ARC" else 0.9,
               label=None, zorder=2 if name == "ARC" else 1)
ax.scatter([1.0], [0.0], s=120, marker="*", c="#27c", edgecolor="k", zorder=4)
ax.annotate("hand-built twin, every neuron\nreplaced by its exact rule", (1.0, 0), (0.45, -6), fontsize=8, arrowprops=dict(arrowstyle="-", lw=0.6))
ax.scatter([], [], s=9, c="#9ab", label="trained networks (21 others)"); ax.scatter([], [], s=9, c="#c33", label="ARC's network")
ax.set_xlim(0.3, 1.02); ax.set_ylim(-80, 5); ax.axhline(0, c="k", lw=0.5)
ax.set_xlabel("variance of the neuron matched by its formula"); ax.set_ylabel("change in accuracy with that neuron\nreplaced by its formula (points)")
ax.legend(fontsize=7.5, loc="lower left", frameon=False)
ax.set_title("Matching a neuron's variance does not mean it can be replaced", fontsize=10)
fig.tight_layout(); fig.savefig("paper/figures/fig_substitution.svg"); fig.savefig("paper/figures/fig_substitution.png", dpi=130)
import numpy as np
pts = [(p[1], p[2] - v["base"]) for k, v in d.items() if k not in ("v5", "v18") for p in v["points"]]
a = np.array(pts); hi = a[:, 0] > 0.95
print(f"trained neurons: {len(a)}; with >95% variance matched: {hi.sum()}, median accuracy change {100 * np.median(a[hi, 1]):.1f} points, share losing >5 points {np.mean(a[hi, 1] < -0.05):.2f}")
