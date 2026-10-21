"""Exact per-pair two-spike accuracy for all 96 copies: how many of the 90 position pairs fail completely?"""
import numpy as np

from algzoo.bench import load_bench
from algzoo.core import example
from algzoo.exact_z2 import exact_z2

names, acc, models = load_bench(("mult",))
rows = []
for nm, m in zip(["example"] + list(names), [example()] + models):
    _, _, _, det = exact_z2(m, return_detail=True)
    a = np.array([v[0] for v in det.values()])
    rows.append((np.sum(a < 0.05), np.sum(a < 0.5), a.min()))
    print(f"{nm:14s} pairs below 0.05: {rows[-1][0]:2d}, below 0.5: {rows[-1][1]:2d}, worst pair {rows[-1][2]:.3f}", flush=True)
R = np.array(rows[1:])
sig = np.array([float(str(n).split("_")[0][4:]) for n in names])
for s in (0.005, 0.01, 0.02):
    k = sig == s
    print(f"σ={s}: copies with at least one pair below 0.05: {np.sum(R[k, 0] > 0)} of 32; median pairs below 0.05 {np.median(R[k, 0]):.0f}, max {R[k, 0].max():.0f}")
np.save("/data/output/research/algzoo/20261016_run2/reports/pair_counts.npy", R)
