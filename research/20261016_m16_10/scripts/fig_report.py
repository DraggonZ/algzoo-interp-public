"""Report figures: variance explained by fixing top-k inputs, region counts by length, accuracy by arrival margin."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch as th

from core import example, forward

OUT = "/data/output/research/algzoo/20261016_m16_10/reports/"
BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2, "text.color": INK, "font.size": 10,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
})

# 1. Share of the variance of correctness explained by fixing the top k values and positions (measured, 20,000 x 256)
k = np.arange(1, 10)
share = np.array([0.040, 0.250, 0.621, 0.757, 0.829, 0.873, 0.903, 0.933, 0.963])
fig, ax = plt.subplots(figsize=(6.2, 3.6))
ax.plot(k, share, color=BLUE, lw=2, marker="o", ms=6, mec=SURF, mew=2, zorder=3)
for kk, s in zip(k, share):
    if kk in (3, 5, 9):
        ax.annotate(f"{s:.0%}", (kk, s), textcoords="offset points", xytext=(-4, 9), ha="center", color=INK, fontsize=9)
ax.set_xticks(k)
ax.set_ylim(0, 1.05)
ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
ax.set_xlabel("number of largest inputs fixed (value and position)")
ax.set_ylabel("variance of correctness explained")
ax.set_title("Every input matters: even with 9 of 10 values fixed,\nthe smallest one still carries 4% of the variance", loc="left", fontsize=11)
fig.tight_layout()
fig.savefig(OUT + "fig_topk_variance.png", dpi=150)

# 2. Linear regions seen in 4 million samples, 16-neuron zoo models, by sequence length (lower bounds; exact 926 at length 3)
n = np.array([2, 3, 4, 5, 6, 8, 10])
seen = np.array([34, 506, 7273, 28561, 69629, 492621, 1823688])
fig, ax = plt.subplots(figsize=(6.2, 3.6))
ax.plot(n, seen, color=BLUE, lw=2, marker="o", ms=6, mec=SURF, mew=2, zorder=3)
ax.scatter([3], [926], s=60, facecolor=SURF, edgecolor=ORANGE, lw=2, zorder=4)
ax.annotate("exact count: 926", (3, 926), textcoords="offset points", xytext=(8, 4), color=INK2, fontsize=9)
ax.annotate("1.8 million and still rising", (10, seen[-1]), textcoords="offset points", xytext=(-150, 4), color=INK2, fontsize=9)
ax.set_yscale("log")
ax.set_xticks(n)
ax.set_xlabel("sequence length (number of inputs)")
ax.set_ylabel("distinct ReLU on/off patterns")
ax.set_title("Why exact computation stops early: linear regions\ngrow 5-10x per extra input (16-neuron models)", loc="left", fontsize=11)
fig.tight_layout()
fig.savefig(OUT + "fig_region_growth.png", dpi=150)

# 3. Accuracy by the margin with which the eventual 2nd max beat the previous record when it arrived
p = example()
dev = "cuda"
N = 4 * 10**6
x = th.randn(N, 10, device=dev, dtype=th.float64, generator=th.Generator(dev).manual_seed(15))
ok = forward(p, x).argmax(-1) == x.argsort(-1)[..., -2]
order = x.argsort(-1, descending=True)
pos1, pos2 = order[:, 0], order[:, 1]
m2 = x.gather(1, pos2[:, None])[:, 0]
rm = th.zeros(N, 11, device=dev, dtype=x.dtype)
r1 = th.full((N, 11), -1e9, device=dev, dtype=x.dtype)
r2 = th.full((N, 11), -1e9, device=dev, dtype=x.dtype)
for t in range(10):
    rm[:, t + 1] = th.maximum(rm[:, t], x[:, t])
    xt = x[:, t]
    r2[:, t + 1] = th.where(xt > r1[:, t], r1[:, t], th.maximum(r2[:, t], xt))
    r1[:, t + 1] = th.maximum(r1[:, t], xt)
early = pos2 < pos1
margin = th.where(early, m2 - rm.gather(1, pos2[:, None])[:, 0], m2 - r2.gather(1, pos2[:, None])[:, 0])
edges = np.array([0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 5])
mids = np.sqrt(np.maximum(edges[:-1], 0.003) * edges[1:])
fig, ax = plt.subplots(figsize=(6.2, 3.8))
for sel, col, lab in [(early & (m2 > 0), BLUE, "2nd max arrived first\n(margin over the running max)"),
                      ((~early) & (m2 > 0), ORANGE, "max arrived first\n(margin over the running 2nd)")]:
    acc = [ok[sel & (margin >= a) & (margin < b)].double().mean().item() for a, b in zip(edges[:-1], edges[1:])]
    ax.plot(mids, acc, color=col, lw=2, marker="o", ms=6, mec=SURF, mew=2, zorder=3)
    ax.plot([], [], color=col, lw=2, marker="o", ms=6, mec=SURF, mew=2, label=lab.replace(chr(10), " "))
ax.axhline(0.9532, color=INK2, lw=1, ls=(0, (3, 3)))
ax.annotate("overall 95.3%", (0.012, 0.9532), textcoords="offset points", xytext=(0, 5), ha="center", color=INK2, fontsize=8.5)
ax.set_xscale("log")
ax.set_ylim(0.55, 1.0)
ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
ax.legend(loc="lower right", frameon=False, fontsize=8.5)
ax.set_xlabel("margin by which the eventual 2nd max beat the record when it arrived")
ax.set_ylabel("accuracy")
ax.set_title("Errors follow close calls at arrival time\n(inputs whose 2nd largest value is positive)", loc="left", fontsize=11)
fig.tight_layout()
fig.savefig(OUT + "fig_arrival_margin.png", dpi=150)
print("saved")
