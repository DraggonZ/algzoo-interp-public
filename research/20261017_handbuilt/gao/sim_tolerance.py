"""Is the account's error harmful because of its size or its pattern? Per neuron: true values plus (a) the account's
own error, (b) the same errors shuffled across lists (same size and step profile, no link to the list), (c) the account's
error scaled by 0.5 and 0.25. Also: the noise size (iid Gaussian, after ReLU clipping) that costs 1 point of accuracy."""
import numpy as np
from simlib import lists, run, score, fit_relu
from sim_groups import account_features

Xf, Xs = lists(101), lists(202)
_, Pf, Hf = run(Xf); Ls, Ps, Hs = run(Xs)
base = score(Xs, Ls, Ls)["acc"]; rng = np.random.default_rng(0)
print(f"base accuracy {base:.3f}")
print(f"{'neuron':>6s} {'acct rms':>8s} {'own err':>7s} {'shuffled':>8s} {'0.5 err':>7s} {'0.25 err':>8s} {'noise for -1pt':>14s}")
for n in range(16):
    sim = fit_relu(account_features(Xf, n), Pf[:, :, n], account_features(Xs, n))[0]
    err = sim - Hs[:, :, n]; rms = np.sqrt((err ** 2).mean())
    acc = lambda v: score(Xs, Ls, run(Xs, {n: v})[0])["acc"]
    own = acc(Hs[:, :, n] + err); shuf = acc(np.maximum(Hs[:, :, n] + err[rng.permutation(len(Xs))], 0))
    half = acc(Hs[:, :, n] + 0.5 * err); quart = acc(Hs[:, :, n] + 0.25 * err)
    tol = None
    for s in (0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2):
        if base - acc(np.maximum(Hs[:, :, n] + s * rng.standard_normal(Hs[:, :, n].shape), 0)) > 0.01:
            tol = s; break
    print(f"{n:6d} {rms:8.3f} {own:7.3f} {shuf:8.3f} {half:7.3f} {quart:8.3f} {('<= %g' % tol) if tol else '> 2':>14s}")
