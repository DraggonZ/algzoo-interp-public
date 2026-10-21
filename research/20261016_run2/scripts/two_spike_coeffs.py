"""Exact comparator coefficients with two spikes: on each angle cell, neuron j at the end equals a·m1 + b·m2.
Mass-weighted mean and spread of (a, b) per neuron and age of the max (δ = 9 - p1), over pairs and cells with
m2 > 0 (2nd largest positive). Rows are neurons 1, 2, 4, 6, 7, 8; values are in units of each neuron's own scale.
Also the share of mass where the neuron is exactly zero.
"""
import numpy as np

from algzoo.core import example
from algzoo.exact_z2 import angle_mass, pair_intervals

params = example()
C = [1, 2, 4, 6, 7, 8]
rows = {j: {d: [] for d in range(10)} for j in C}
ops = [0]
for p1 in range(10):
    for p2 in range(10):
        if p1 == p2:
            continue
        for a, b, A in pair_intervals(params, p1, p2, ops):
            lo = max(a, 0.0)                        # m2 > 0  <=>  θ in (0, π/4)
            if b <= lo:
                continue
            m = angle_mass(lo, b)
            for j in C:
                rows[j][9 - p1].append((m, A[j, 0], A[j, 1], 9 - p2))
np.set_printoptions(precision=2, suppress=True)
summary = np.full((len(C), 10, 6), np.nan)       # neuron, δ, (a mean, a p10, a p90, b mean, b p10, b p90)
for j in C:
    print(f"\nneuron {j}: age of max δ: mass-weighted a (coef on max), b (coef on 2nd); [10th-90th percentile]; share zero")
    for d in range(10):
        R = np.array(rows[j][d])
        if not len(R):
            continue
        w = R[:, 0] / R[:, 0].sum()
        def wq(v, q):
            o = np.argsort(v); c = np.cumsum(w[o]); return v[o][np.searchsorted(c, q)]
        zero = w[(np.abs(R[:, 1]) < 1e-12) & (np.abs(R[:, 2]) < 1e-12)].sum()
        summary[C.index(j), d] = [np.sum(w * R[:, 1]), wq(R[:, 1], .1), wq(R[:, 1], .9), np.sum(w * R[:, 2]), wq(R[:, 2], .1), wq(R[:, 2], .9)]
        print(f"  δ={d}: a {np.sum(w * R[:, 1]):6.2f} [{wq(R[:, 1], .1):6.2f},{wq(R[:, 1], .9):6.2f}]  b {np.sum(w * R[:, 2]):6.2f} [{wq(R[:, 2], .1):6.2f},{wq(R[:, 2], .9):6.2f}]  zero {zero:.2f}")

np.savez("/data/output/research/algzoo/20261016_run2/reports/two_spike_coeffs.npz", summary=summary, neurons=np.array(C))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "text.color": INK, "font.size": 9,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8})
names = {1: "n1: max without x(t-3), minus newest", 2: "n2: running max", 4: "n4: new-max jump", 6: "n6: max without x(t-2), minus newest",
         7: "n7: max minus newest", 8: "n8: roughly 2nd max minus newest"}
fig, axes = plt.subplots(2, 3, figsize=(9.6, 5.2), sharex=True, sharey=True, constrained_layout=True)
d = np.arange(10)
for ax, j in zip(axes.flat, [2, 7, 6, 1, 8, 4]):
    S = summary[C.index(j)]
    for k, col, lab in [(0, BLUE, "coefficient on the max (a)"), (3, ORANGE, "coefficient on the 2nd (b)")]:
        ax.fill_between(d, S[:, k + 1], S[:, k + 2], color=col, alpha=0.15, lw=0)
        ax.plot(d, S[:, k], color=col, lw=2, marker="o", ms=3.5, label=lab)
    ax.axhline(0, color=INK2, lw=0.8)
    ax.set_title(names[j], loc="left", fontsize=9)
for ax in axes[1]:
    ax.set_xlabel("age of the max at the readout (steps)")
for ax in axes[:, 0]:
    ax.set_ylabel("exact coefficient")
axes[0, 0].legend(frameon=False, fontsize=8, loc="lower left")
fig.suptitle("With two spikes, each comparator neuron is exactly a*max + b*2nd on every linear cell\n(lines: probability-weighted mean over cells; bands: 10-90% range; 2nd largest positive)",
             x=0.01, ha="left", fontsize=10)
fig.savefig("/data/output/research/algzoo/20261016_run2/reports/fig_comparator_coeffs.png", dpi=150)
