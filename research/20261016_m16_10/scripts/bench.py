"""Score accuracy estimators on the noisy-model benchmark against random sampling.

An estimator is a function params -> (estimate, ops), where ops counts multiply-adds.
Sampling with N samples costs N * FWD_OPS and has expected squared error p(1-p)/N.
"""
import numpy as np

BENCH = "/data/output/research/algzoo/20261016_m16_10/reports/benchmark.npz"
FWD_OPS = 10 * 16 * 17 + 16 * 10   # 2,880 multiply-adds per forward pass of M16,10


def load_bench(groups=("mult",)):
    d = np.load(BENCH)
    names = d["names"]
    sel = np.array([any(str(n).startswith(g) for g in groups) for n in names])
    models = [(d["w_hi"][i], d["w_hh"][i], d["w_oh"][i]) for i in np.flatnonzero(sel)]
    return names[sel], d["acc"][sel], models


def score(estimator, groups=("mult",), verbose=True):
    names, acc, models = load_bench(groups)
    est, ops = [], []
    for m in models:
        e, o = estimator(m)
        est.append(e)
        ops.append(o)
    est, ops = np.array(est), np.array(ops, dtype=float)
    mse = np.mean((est - acc) ** 2)
    n_equiv = ops.mean() / FWD_OPS
    mse_sampling = np.mean(acc * (1 - acc)) / n_equiv
    out = {
        "rmse": np.sqrt(mse),
        "bias": np.mean(est - acc),
        "ops": ops.mean(),
        "samples_equiv": n_equiv,
        "rmse_sampling_same_ops": np.sqrt(mse_sampling),
        "mse_ratio": mse / mse_sampling,     # < 1 beats sampling
        "corr": np.corrcoef(est, acc)[0, 1] if len(acc) > 2 else np.nan,
    }
    if verbose:
        print(
            f"RMSE {out['rmse']:.4f} (bias {out['bias']:+.4f}, corr {out['corr']:.3f}) at {out['ops']:.3g} ops"
            f" = {n_equiv:.3g} samples; sampling RMSE {out['rmse_sampling_same_ops']:.4f};"
            f" MSE ratio {out['mse_ratio']:.3g}"
        )
    return out, est, acc, names
