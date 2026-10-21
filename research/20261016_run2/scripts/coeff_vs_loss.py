"""Across the 96 copies: does the exact two-spike loss at each age of the max track the copies' exact comparator
coefficient shifts at that age? (Spearman correlations over copies x ages; no fitting to true accuracies.)"""
import numpy as np
from scipy.stats import spearmanr

from algzoo.bench import load_bench
from algzoo.core import example
from coeff_compare import table, C

names, acc, models = load_bench(("mult",))
Tb, ab = table(example())
rows = []
for m in models:
    Tc, ac = table(m)
    da = np.abs(Tc[:, :, 0] - Tb[:, :, 0]) / np.maximum(np.abs(Tb[:, :, 0]), 1.0)      # [neuron, age]
    db = np.abs(Tc[:, :, 1] - Tb[:, :, 1])
    rows.append((ab - ac, da, db))
loss = np.concatenate([r[0] for r in rows])
for j, n in enumerate(C):
    a = np.concatenate([r[1][j] for r in rows])
    b = np.concatenate([r[2][j] for r in rows])
    print(f"n{n}: Spearman(loss at age, |Δa| at age) {spearmanr(loss, a)[0]:+.2f}; with |Δb| {spearmanr(loss, b)[0]:+.2f}")
allshift = np.concatenate([r[1].sum(0) + r[2].sum(0) / 10 for r in rows])
print(f"all comparators summed: Spearman {spearmanr(loss, allshift)[0]:+.2f}")
per_copy = np.array([r[0].mean() for r in rows])
shift_copy = np.array([r[1][[C.index(8), C.index(1)]].sum() for r in rows])
print(f"per copy: Spearman(two-spike loss, total |Δa| of n8 and n1) {spearmanr(per_copy, shift_copy)[0]:+.2f}")
