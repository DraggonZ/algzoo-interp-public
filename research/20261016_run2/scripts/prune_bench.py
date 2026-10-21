"""Budgeted (pruned) exact enumeration on the noisy M16,4 benchmark: squared error against sampling at equal cost."""
import os
import sys

import numpy as np

os.environ.setdefault("OMP_NUM_THREADS", "1")
import exact_nd as E  # noqa: E402

D = dict(np.load("/data/output/research/algzoo/20261016_run2/reports/bench_16_4.npz"))
names, truth = D["names"], D["acc"]
sig = np.array([0.0 if str(s) == "base" else float(str(s).split("_")[0][4:]) for s in names])
FWD = 4 * (256 + 32)
out = {}
CORRECT = sys.argv[1] == "corrected"
for eps in [float(e) for e in sys.argv[2:]]:
    est, ops = [], []
    for i in range(len(names)):
        a, st = E.exact_accuracy((D["w_hi"][i], D["w_hh"][i], D["w_oh"][i]), prune=eps)
        est.append(st["corrected"] if CORRECT else a)
        ops.append(st["ops"])
    est, ops = np.array(est), np.array(ops, dtype=float)
    out[eps] = (est, ops)
    for s in (0.005, 0.01, 0.02):
        k = sig == s
        p = truth[k]
        mse = np.mean((est[k] - p) ** 2)
        B = ops[k].mean() / FWD
        mse_s = np.mean(p * (1 - p)) / B
        print(f"{'corrected' if CORRECT else 'plain'} prune {eps:.0e} σ={s}: RMSE {np.sqrt(mse):.2e} (bias {np.mean(est[k] - p):+.2e}); cost {B:.0f} samples;"
              f" sampling RMSE {np.sqrt(mse_s):.2e}; squared-error ratio {mse / mse_s:.3f}", flush=True)
    k = sig > 0
    mse = np.mean((est[k] - truth[k]) ** 2)
    B = ops[k].mean() / FWD
    mse_s = np.mean(truth[k] * (1 - truth[k])) / B
    print(f"{'corrected' if CORRECT else 'plain'} prune {eps:.0e} all 96: RMSE {np.sqrt(mse):.2e}, cost {B:.0f} samples, sampling RMSE {np.sqrt(mse_s):.2e}, ratio {mse / mse_s:.3f}", flush=True)
np.save(f"/data/output/research/algzoo/20261016_run2/reports/prune_bench_{sys.argv[1]}_{'_'.join(sys.argv[2:])}.npy", {str(k): v for k, v in out.items()}, allow_pickle=True)
