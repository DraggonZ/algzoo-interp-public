"""Exact accuracy (exact_nd.py) on every model of a length-n noisy benchmark, scored against sampling.

Sampling at equal compute: N = ops / (n * (d^2 + 2d)) samples, expected squared error p(1-p)/N.
The reference accuracies have their own sampling error p(1-p)/n_truth; it is reported, not subtracted.
"""
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import exact_nd as E  # noqa: E402

n = int(sys.argv[1])
workers = int(sys.argv[2]) if len(sys.argv) > 2 else 5
TAG = sys.argv[3] if len(sys.argv) > 3 else ""
D = dict(np.load(f"/data/output/research/algzoo/20261016_run2/reports/bench_16_{n}.npz"))


def job(i):
    t0 = time.time()
    acc, st = E.exact_accuracy((D["w_hi"][i], D["w_hh"][i], D["w_oh"][i]))
    return i, acc, st["ops"], st["regions"], time.time() - t0


if __name__ == "__main__":
    names, truth, n_truth = D["names"], D["acc"], int(D["n_truth"])
    subset = [int(i) for i in sys.argv[4].split(",")] if len(sys.argv) > 4 else list(range(len(names)))
    with Pool(workers) as pool:
        res = sorted(pool.imap_unordered(job, subset))
    names, truth = names[subset], truth[subset]
    est = np.array([r[1] for r in res])
    ops = np.array([r[2] for r in res], dtype=float)
    regions = np.array([r[3] for r in res])
    secs = np.array([r[4] for r in res])
    d = D["w_hh"].shape[1]
    fwd = n * (d * d + 2 * d)
    np.savez(f"/data/output/research/algzoo/20261016_run2/reports/exact_bench_16_{n}{TAG}.npz", names=names, truth=truth, est=est,
             ops=ops, regions=regions, secs=secs, fwd_ops=fwd)
    for i in range(len(names)):
        se = np.sqrt(truth[i] * (1 - truth[i]) / n_truth)
        print(f"{names[i]:14s} exact {est[i]:.8f} truth {truth[i]:.8f} ± {se:.1e} z {(est[i] - truth[i]) / se:+.2f}"
              f" regions {regions[i]} ops {ops[i]:.3g} ({secs[i]:.0f}s)")
    sig = np.array([0.0 if str(s) == "base" else float(str(s).split("_")[0][4:]) for s in names])
    for s in (0.005, 0.01, 0.02):
        k = sig == s
        if not k.any():
            continue
        p = truth[k]
        mse = np.mean((est[k] - p) ** 2)
        ref = np.mean(p * (1 - p)) / n_truth
        N = ops[k].mean() / fwd
        mse_s = np.mean(p * (1 - p)) / N
        print(f"σ={s}: accuracy {p.min():.4f}-{p.max():.4f}; exact RMSE {np.sqrt(mse):.2e} (reference's own error {np.sqrt(ref):.2e});"
              f" cost {ops[k].mean():.3g} ops = {N:.3g} samples; sampling RMSE at that cost {np.sqrt(mse_s):.2e};"
              f" MSE ratio {mse / mse_s:.2e}")
    z = (est - truth) / np.sqrt(truth * (1 - truth) / n_truth)
    print(f"all {len(z)}: mean z {z.mean():+.2f}, SD of z {z.std():.2f}, max |z| {np.abs(z).max():.2f}")
