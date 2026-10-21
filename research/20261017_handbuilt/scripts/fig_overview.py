"""Overview diagram: one list, step by step, showing which votes exist and which are hidden.
List: 0.3, 1.0, 0.2, -0.5, 0.5, 1.5, then four negative numbers. Cancelling-neuron activity and final scores checked on
the trained network (cancellers zero at steps 0, 1 and 5, on at 2-4 and from 6; answer position 1, score 28.7)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
xs = [0.3, 1.0, 0.2, -0.5, 0.5, 1.5, -0.4]
fig, ax = plt.subplots(figsize=(8.0, 3.7)); ax.axis("off"); ax.set_xlim(-1.9, 7.6); ax.set_ylim(-0.1, 4.6)
for t, x in enumerate(xs):
    ax.text(t, 4.3, f"{x:+.1f}", ha="center", fontsize=10, weight="bold"); ax.text(t, 4.02, f"pos {t}", ha="center", fontsize=7, c="#666")
ax.text(6.9, 4.3, "...", fontsize=10)
rows = {"largest so far": 3.45, "gate, one step later": 2.9, "vote for the 0.3": 2.35, "vote for the 1.0": 1.8, "vote for the 0.5": 1.25, "cancelling neurons": 0.7}
for k, y in rows.items():
    ax.text(-1.85, y, k, va="center", fontsize=8)
for t, m in enumerate([0.3, 1.0, 1.0, 1.0, 1.0, 1.5, 1.5]):
    ax.text(t, rows["largest so far"], f"{m:.1f}", ha="center", va="center", fontsize=8)
for t, g in enumerate(["", "keep 0.3", "keep 1.0", "forget 0.2", "forget", "keep 0.5", "keep 1.5"]):
    ax.text(t, rows["gate, one step later"], g, ha="center", va="center", fontsize=7, c="#173" if g.startswith("keep") else "#a33")
def box(t0, t1, y, text, fc, ec="k", ls="-"):
    ax.add_patch(FancyBboxPatch((t0 - 0.42, y - 0.22), t1 - t0 + 0.84, 0.44, boxstyle="round,pad=0.02", fc=fc, ec=ec, ls=ls, lw=0.8))
    ax.text((t0 + t1) / 2, y, text, ha="center", va="center", fontsize=7.5)
box(1, 6, rows["vote for the 0.3"], "shows once the 1.0 arrives (final score 11.6)", "#e8f0ff")
box(1, 4, rows["vote for the 1.0"], "hidden while it is the largest", "#eef", ls="--")
box(5, 6, rows["vote for the 1.0"], "shows: score 28.7, wins", "#cdf")
box(4, 6, rows["vote for the 0.5"], "kept, but halved by the 1.5 right after it", "#efe")
box(2, 4, rows["cancelling neurons"], "on: cancel the 1.0's vote", "#fdd")
for t in (1, 5):
    ax.text(t, rows["cancelling neurons"], "silenced", ha="center", va="center", fontsize=7, c="#a33")
box(6, 6, rows["cancelling neurons"], "rebuilt", "#fdd")
ax.text(2.9, 0.1, "A vote is hidden while its number is the largest. The next bigger number unhides it, exactly when it becomes the second.", fontsize=7.5, ha="center", c="#333")
fig.tight_layout(); fig.savefig("paper/figures/fig0_overview.svg"); fig.savefig("paper/figures/fig0_overview.png", dpi=130)
