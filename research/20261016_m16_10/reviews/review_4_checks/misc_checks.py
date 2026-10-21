"""Review 4 checks (cheap): seed 3 = best-of-five file at length 4; uncounted arithmetic in exact_nd.py on M16,4;
budgeted-mode ratios on the trained seeds with full precision; near-tie per-σ ratios and the post-hoc 'zero' cases."""
import os

os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np  # noqa: E402
import torch as th  # noqa: E402

import algzoo.exact_nd as E  # noqa: E402
from algzoo.core import accuracy, load, zoo  # noqa: E402

REP = "/data/output/research/algzoo/20261016_run2/reports/"

# 1. is the M16,4 best-of-five file zoo seed 3?
bo5 = load("2nd_argmax_16_4_1073741824_bo5.pth")
for s in range(5):
    same = all(np.array_equal(a, b) for a, b in zip(bo5, zoo(16, 4, s)))
    print(f"M16,4 bo5 identical to seed {s}: {same}")

# 2. arithmetic exact_nd.py does but does not count (qhull aside): Gram matrix + determinant per simplicial piece,
#    SVD + mean per non-simplicial cone. Counted as n^3 + n^3 (Gram, det) and 4 n^3 (SVD), a rough lower bound.
calls = {"simp": 0, "cone": 0}
_sm, _cm = E.simplicial_measure, E.cone_measure


def sm(V, ops, max_angle=None, q=None):
    calls["simp"] += 1
    return _sm(V, ops, max_angle, q)


def cm(R, ops):
    if R.shape[0] > R.shape[1]:
        calls["cone"] += 1
    return _cm(R, ops)


E.simplicial_measure, E.cone_measure = sm, cm
for lab, kw in [("full", {}), ("prune 3e-6", {"prune": 3e-6}), ("prune 1e-5", {"prune": 1e-5})]:
    calls.update(simp=0, cone=0)
    a, st = E.exact_accuracy(bo5, **kw)
    n = 4
    extra = calls["simp"] * 2 * n**3 + calls["cone"] * 4 * n**3
    print(f"M16,4 {lab}: counted {st['ops']:.4g} = {st['ops'] / 1152:.0f} samples; simplicial pieces {calls['simp']}, "
          f"non-simplicial cones {calls['cone']} (= qhull calls); uncounted Gram/det/SVD about {extra:.3g} = {extra / 1152:.0f} samples "
          f"(+{100 * extra / st['ops']:.0f}%)")
E.simplicial_measure, E.cone_measure = _sm, _cm

# 3. budgeted mode on the trained seeds, full precision (same seeds as prune_seeds.py)
print("budgeted mode, trained seeds (prune_seeds.py settings):")
for s in range(5):
    p = zoo(16, 4, s)
    truth, nn = accuracy(p, 2**31, batch=2**24, seed=9500 + s)
    for eps in (3e-6, 1e-5):
        a, st = E.exact_accuracy(p, prune=eps)
        B = st["ops"] / 1152
        mse_s = truth * (1 - truth) / B
        err = st["corrected"] - truth
        print(f"  seed {s} prune {eps:.0e}: truth {truth:.7f} error {err:+.3e} cost {B:.0f} sampling RMSE {np.sqrt(mse_s):.3e} "
              f"ratio {err**2 / mse_s:.3f}")

# 4. near-tie pilot: per-σ squared-error ratios; what the scan says on cases the post-hoc rule zeroes
z = np.load(REP + "neartie_score_zero.npz")
sig = np.array([float(str(n).split("_")[0][4:]) for n in z["names"]])
for s in (0.005, 0.01, 0.02):
    k = sig == s
    B = z["ops"][k].mean() / 2880
    mse = np.mean((z["est"][k] - z["acc"][k]) ** 2)
    mse_s = np.mean(z["acc"][k] * (1 - z["acc"][k])) / B
    print(f"near-tie zero rule σ={s}: RMSE {np.sqrt(mse):.4f}, cost {B:.0f}, squared-error ratio {mse / mse_s:.1f}")
k = sig > 0
B = z["ops"][k].mean() / 2880
mse = np.mean((z["est"][k] - z["acc"][k]) ** 2)
print(f"near-tie zero rule pooled 96: RMSE {np.sqrt(mse):.4f}, cost {B:.0f}, ratio {mse / (np.mean(z['acc'][k] * (1 - z['acc'][k])) / B):.1f}")
c = np.load(REP + "neartie_check.npz")
w, ws, m = c["w"], c["ws"], c["mass"]
inf_l = ~np.isfinite(w)
tot = m.sum()
print(f"near-tie check: linear width infinite (zeroed by the post-hoc rule) on {(m * inf_l).sum() / tot:.4f} of case mass; "
      f"of that, scan also fails over the whole 4-unit range {(m * (inf_l & ~np.isfinite(ws))).sum() / (m * inf_l).sum():.2f}, "
      f"scan finds a finite band {(m * (inf_l & np.isfinite(ws) & (ws > 0))).sum() / (m * inf_l).sum():.2f}, scan zero "
      f"{(m * (inf_l & (ws == 0))).sum() / (m * inf_l).sum():.2f}")
both_zero = (w < 1e-3) & (ws < 1e-3)
print(f"near-tie check: both widths zero on {(m * both_zero).sum() / tot:.3f} of case mass")
th.cuda.empty_cache()
