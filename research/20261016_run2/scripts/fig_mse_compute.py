"""ARC's view: squared error relative to sampling at equal compute, sampling-free methods (logged results only).

y = squared error ÷ sampling's expected squared error at the same compute, so sampling is the line y = 1.
Left: length 4 (M16,4), trained seeds (primary) and noisy copies per σ. Right: length 10 (M16,10), noisy copies pooled
over σ, with the per-σ spread for the two best estimators.
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

R = "/data/output/research/algzoo/20261016_run2/reports/"
ORANGE, INK, INK2, GRID, SURF = "#eb6834", "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
BLUES = ["#86b6ea", "#3f86d9", "#1a4f93"]   # σ = 0.005, 0.01, 0.02 (ordered, one hue light to dark)
plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "text.color": INK, "font.size": 9.5,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8})
fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), constrained_layout=True)

# ---- length 4 ----
ax = axes[0]
ax.axhline(1, color=INK2, lw=2)
ax.text(2.1e4, 1.15, "random sampling", fontsize=8, color=INK2)
# noisy copies per σ, thresholds 1e-4, 3e-5, 1e-5, 3e-6 (prune_bench_recount_a/b.log)
copies = {0.005: [(31697, 5.137), (42830, 1.632), (54866, 0.439), (67750, 0.100)],
          0.01: [(34334, 5.529), (46595, 1.244), (59358, 0.349), (72837, 0.079)],
          0.02: [(38130, 7.076), (52172, 2.051), (66179, 0.526), (82050, 0.082)]}
D = np.load(R + "exact_bench_16_4_recount.npz")
names = np.array([str(n) for n in D["names"]])
for c, (s, pts) in zip(BLUES, copies.items()):
    ax.plot([x for x, _ in pts], [y for _, y in pts], color=c, lw=1.5, marker="o", ms=5, mec=SURF, mew=1.2,
            label=f"budgeted, noisy copies σ={s}")
    k = np.array([n.startswith(f"mult{s}_") for n in names])
    p = D["truth"][k]
    cost = D["ops"][k].mean() / D["fwd_ops"]
    ratio = np.mean((D["est"][k] - p) ** 2) / (np.mean(p * (1 - p)) / cost)
    ax.scatter([cost], [ratio], s=45, color=c, edgecolor=SURF, lw=1.2, marker="D", zorder=4)
ax.annotate("full exact, per σ (error about the size of\nthe reference's own sampling error)", (1.48e5, 1.1e-4),
            textcoords="offset points", xytext=(-205, 12), fontsize=8, color=INK2)
# trained seeds 0-4 at thresholds 3e-6 and 1e-5 (prune_seeds_recount.log; ratios at full precision, review 4)
seeds = {3e-6: ([69407, 69850, 82262, 63358, 68622], [0.196, 0.178, 0.329, 0.123, 0.239]),
         1e-5: ([55126, 57441, 64509, 52193, 54385], [0.527, 0.624, 0.996, 0.400, 0.649])}
for j, (th, (cs, rs)) in enumerate(seeds.items()):
    ind = [0, 1, 2, 4]
    ax.scatter([cs[i] for i in ind], [rs[i] for i in ind], s=55, color=ORANGE, edgecolor=SURF, lw=1.2, marker="^", zorder=5,
               label="budgeted, trained seeds 0, 1, 2, 4" if j == 0 else None)
    ax.scatter([cs[3]], [rs[3]], s=55, facecolor=SURF, edgecolor=ORANGE, lw=1.5, marker="^", zorder=5,
               label="budgeted, seed 3 (= base; correction designed on it)" if j == 0 else None)
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlim(2e4, 2.2e5)
ax.set_ylim(5e-5, 15)
ax.set_xlabel("compute (M16,4 forward passes' worth of counted multiply-adds;\nqhull and per-piece Gram/SVD work not counted)")
ax.set_ylabel("squared error ÷ sampling's at equal compute")
ax.set_title("Length 4 (M16,4), exact computation, not an explanation:\nahead of sampling above about 55,000-65,000 samples' worth", loc="left", fontsize=10)
ax.legend(frameon=False, fontsize=7.5, loc="lower left", bbox_to_anchor=(0, 0.33))
ax.set_xticks([2e4, 3e4, 5e4, 1e5, 2e5], ["20k", "30k", "50k", "100k", "200k"])
ax.minorticks_off()

# ---- length 10 ----
B = np.load("/data/output/research/algzoo/20261016_m16_10/reports/benchmark.npz")
kk = np.array([str(n).startswith("mult") for n in B["names"]])
p10 = B["acc"][kk]
pq10 = np.mean(p10 * (1 - p10))
ax = axes[1]
ax.axvspan(1e3, 1e5, color=BLUES[1], alpha=0.07, lw=0)
ax.text(1.15e3, 0.6, "the note's 1,000-100,000-sample window", fontsize=7.5, color=INK2)
ax.axhline(1, color=INK2, lw=2)
ax.axhline(3, color=INK2, lw=1, ls=":")
ax.text(12, 1.15, "random sampling", fontsize=8, color=INK2)
ax.text(12, 3.4, "3× sampling (run-2 milestone)", fontsize=8, color=INK2)
# (label, cost in samples, pooled RMSE, per-σ squared-error ratios or None)
est = [("two-spike accuracy Z_2 (exact)", 2890, 0.122, None),
       ("per-cell margin (review 1)", 2160, 0.066, (7.1, 48.8, 91.3)),
       ("single-input rule, dense quadrature", 418320, 0.025, None),
       ("", 495, 0.149, None), ("", 1844, 0.109, None), ("single-input rule, cheap quadrature (3 settings; best 72×)", 10660, 0.0761, None),
       ("linearized near-tie width (post-hoc rule)", 12900, 0.036, (35.0, 67.6, 162.8))]
ratio = [r ** 2 / (pq10 / c) for _, c, r, _ in est]
ax.scatter([e[1] for e in est], ratio, s=40, color=ORANGE, edgecolor=SURF, lw=1.2, zorder=4, label="sampling-free estimators, pooled over σ")
for (lab, c, r, per), y in zip(est, ratio):
    if per:
        ax.plot([c, c], [min(per), max(per)], color=ORANGE, lw=1, alpha=0.6, zorder=3)
        ax.scatter([c] * 3, per, s=12, color=ORANGE, marker="_", zorder=3)
offsets = {"two-spike accuracy Z_2 (exact)": (7, -9), "per-cell margin (review 1)": (-88, -26),
           "single-input rule, dense quadrature": (-175, -3), "single-input rule, cheap quadrature (3 settings; best 72×)": (-215, 9),
           "linearized near-tie width (post-hoc rule)": (6, -12)}
for (lab, c, r, per), y in zip(est, ratio):
    if lab:
        ax.annotate(lab.replace("margin (", "margin\n(") + (f" {y:.0f}×" if lab.startswith("per-cell") else ""), (c, y), textcoords="offset points",
                    xytext=offsets[lab], fontsize=7.5, color=INK2)
ax.plot([], [], color=ORANGE, lw=1, alpha=0.6, label="range over the three σ levels")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlim(10, 1e6)
ax.set_ylim(0.5, 1e4)
ax.set_xlabel("compute (M16,10 forward passes' worth of counted multiply-adds)")
ax.set_title("Length 10 (M16,10): the best sampling-free estimators\nare about 60-70× worse than sampling (pooled over σ)", loc="left", fontsize=10)
ax.legend(frameon=False, fontsize=7.5, loc="upper left")
fig.savefig(R + "fig_mse_compute.png", dpi=150)
print("saved; length-10 pooled ratios:", [round(x, 1) for x in ratio])
