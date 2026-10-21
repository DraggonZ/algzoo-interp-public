"""Run 3's band estimator, fixed background against the conditional mean, on ARC's network and the 32 others, scored
against the 10M-list truths in across.npy. Cost in network runs."""
import numpy as np
from algzoo.core import load, example
import bands_cond as bc
tru = np.load("across.npy")[:, 0]
nets = [example()] + [load(f"/data/output/research/algzoo/20261016_run3/models/2nd_argmax_16_10_1073741824_{s}.pth") for s in list(range(100, 116)) + list(range(200, 216))]
for cond in (False, True):
    bc.COND = cond; err = []; cost = []
    for p in nets:
        e, ops, _ = bc.estimate(p); err.append(100 * (e - tru[len(err)])); cost.append(ops / (10 * (16 * 16 + 16) + 160))
    err = np.array(err)
    print(f"background {'conditional mean' if cond else 'fixed -0.46'}: RMSE {np.sqrt((err**2).mean()):.2f} points over 33 networks, mean error {err.mean():+.2f}, ARC {err[0]:+.2f}; cost about {np.mean(cost):.0f} network runs")
