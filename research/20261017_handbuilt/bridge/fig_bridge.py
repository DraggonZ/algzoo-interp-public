"""Summary figure of the bridge study: (a) input weights of the trained network and of the matched hand-built network
(gains 10, exact relabelling and rescaling); (b) variance of each trained memory neuron matched linearly by v18's slots
and store, by plain input features, and by v5 (memory_as_slots2.log); (c) accuracy against partial switching of the
hiding group, v18 over clock gains (g_sweep.log) and the trained network (partial_switch_scale.log)."""
import re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from algzoo.core import example
T = np.asarray(example()[0], float); H = np.load("v18ts_matched2.npz")["wi"]
fig, ax = plt.subplots(1, 3, figsize=(15, 4.4), dpi=200)
x = np.arange(16)
ax[0].bar(x - 0.2, T, 0.4, color="#333", label="trained"); ax[0].bar(x + 0.2, H, 0.4, color="#6a3d9a", label="hand-built, matched")
ax[0].set_xticks(x); ax[0].set_xlabel("neuron"); ax[0].set_ylabel("input weight"); ax[0].legend(frameon=False, fontsize=8)
ax[0].set_title("(a) Input weights after relabelling and rescaling\n(same function; hand-built at gains 10)", fontsize=10)
rows = [l.split("|") for l in open("memory_as_slots2.log").read().splitlines()[1:]]
d = {int(r[0]): [float(v) for v in r[1:]] for r in rows}
mem = [10, 11, 0, 12, 13, 14, 15, 3]
for i, (k, c, lab) in enumerate([(1, "#6a3d9a", "v18 slots + store"), (2, "#bc80bd", "v5 (other memory design)"), (3, "#bbb", "21 input features")]):
    ax[1].bar(np.arange(8) + (i - 1) * 0.27, [d[n][k] for n in mem], 0.27, color=c, label=lab)
ax[1].set_xticks(range(8)); ax[1].set_xticklabels([str(n) for n in mem]); ax[1].set_ylim(0.3, 1.02)
ax[1].set_xlabel("trained memory neuron"); ax[1].set_ylabel("variance matched (held-out lists)"); ax[1].legend(frameon=False, fontsize=8, loc="lower left")
ax[1].set_title("(b) The trained memory is close to a linear re-coding\nof the hand-built delay line (except 3 and 15)", fontsize=10)
g = [(int(m.group(1)), float(m.group(2)), float(m.group(3))) for m in re.finditer(r"g =\s*(\d+): accuracy ([0-9.]+); partly off one step after a record ([0-9.]+)", open("g_sweep.log").read())]
ax[2].plot([100 * p for _, _, p in g], [100 * a for _, a, _ in g], "o-", color="#6a3d9a", label="hand-built v18, clock gain 10 to 300")
for gg, a, p in g:
    if gg in (10, 50, 300): ax[2].annotate(f"g={gg}", (100 * p, 100 * a), textcoords="offset points", xytext=(6, -10), fontsize=8)
tr = float(re.search(r"trained\s+1 step\(s\) after: partly off ([0-9.]+)", open("partial_switch_scale.log").read()).group(1))
ax[2].plot([100 * tr], [95.3], "s", color="#333", ms=8, label="trained network")
ax[2].set_xscale("log"); ax[2].set_xlabel("partial switches one step after a record (% of numbers, log scale)")
ax[2].set_ylabel("accuracy (%)"); ax[2].legend(frameon=False, fontsize=8, loc="lower left")
ax[2].set_title("(c) Partial switching costs the hand-built design;\nthe trained network does it most of the time", fontsize=10)
for a_ in ax: a_.spines[["top", "right"]].set_visible(False)
fig.tight_layout(); fig.savefig("fig_bridge.png", facecolor="white")
