"""The mean-background estimate (full_mean.py, grid 56 x 96 x 96) against true accuracy (20M lists) on ARC's network
and the 32 trained networks of run 3 (seeds 100-115 were trained after run 3's estimate was frozen)."""
import numpy as np, torch
from algzoo.core import load, example
import adf, full_mean, truth
rows = []
nets = [("ARC", example())] + [(str(s), load(f"/data/output/research/algzoo/20261016_run3/models/2nd_argmax_16_10_1073741824_{s}.pth"))
                               for s in list(range(100, 116)) + list(range(200, 216))]
for name, p in nets:
    W = tuple(torch.tensor(np.asarray(q, float), device="cuda", dtype=torch.float64) for q in p)
    t = truth.true_acc(W, n=10_000_000, seed=1); e, _ = full_mean.estimate(W, 56, 96)
    rows.append((name, t, e)); print(f"{name:>4s}: true {100*t:6.2f}%  estimate {100*e:6.2f}%  error {100*(e-t):+.2f}", flush=True)
err = np.array([r[2] - r[1] for r in rows]) * 100
print(f"all {len(rows)}: RMSE {np.sqrt((err**2).mean()):.2f} points, mean error {err.mean():+.2f}, largest {np.abs(err).max():.2f}")
print(f"seeds 100-115 only: RMSE {np.sqrt((err[1:17]**2).mean()):.2f}")
np.save("across.npy", np.array([(r[1], r[2]) for r in rows]))
