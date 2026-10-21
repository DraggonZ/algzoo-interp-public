"""Cost of exact accuracy computation by sequence length (in samples' worth of multiply-adds), against ARC's window."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


BLUE, ORANGE, VIOLET = "#2a78d6", "#eb6834", "#7c4dbd"
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "text.color": INK, "font.size": 10,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8})
# costs from this script's first run (same code): samples' worth = multiply-adds / (n (d^2 + 2d))
rows = {(8, 3): 992, (8, 4): 1.23e5, (16, 3): 909, (16, 4): 1.42e5, (32, 3): 1057, (32, 4): 2.38e5}   # cost_recount_table.log
# length 5 from the logged runs, which did not count the adjacency test (true cost higher): M8,5 1.74e10, M16,5 7.25e10
rows[(8, 5)] = 1.74e10 / (5 * (64 + 16))
rows[(16, 5)] = 7.25e10 / (5 * (256 + 32))
fig, ax = plt.subplots(figsize=(6.4, 3.8), constrained_layout=True)
ax.axhspan(1e3, 1e5, color=BLUE, alpha=0.08, lw=0)
ax.annotate("ARC's window: match sampling\nfrom 1,000 to 100,000 samples", (4.25, 1.4e3), color=INK2, fontsize=8.5, va="bottom")
for d, col in [(8, VIOLET), (16, BLUE), (32, ORANGE)]:
    ns = [n for n in (3, 4, 5) if (d, n) in rows]
    ax.plot(ns, [rows[(d, n)] for n in ns], color=col, lw=2, marker="o", ms=6, mec=SURF, mew=1.5, label=f"{d} neurons")
ax.set_yscale("log")
ax.set_xticks([3, 4, 5])
ax.set_xlim(2.8, 5.3)
ax.set_xlabel("sequence length (number of inputs)")
ax.set_ylabel("cost of exact accuracy (samples' worth of\ncounted multiply-adds; qhull not counted)")
ax.legend(frameon=False, fontsize=9, loc="upper left")
ax.set_title("Full exact accuracy costs about 1,000 samples' worth at length 3 and\n120,000-240,000 at length 4; it grows 150x or more per extra input\n(length-5 points omit the adjacency test, so they are lower bounds)", loc="left", fontsize=9.5)
fig.savefig("/data/output/research/algzoo/20261016_run2/reports/fig_ladder.png", dpi=150)
print("saved")
