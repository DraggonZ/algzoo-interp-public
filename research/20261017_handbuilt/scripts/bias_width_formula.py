"""Predict each model's error rate from its mean unsigned tie biases, with no fitted constants:
error ~ (inputs with at most one positive) x (8/9) + c23 * w23 + c12 * w12.
c23, c12: density of near-ties among random lists, per unit of relative gap: the rate at which
P(second - third < d * second) and P(largest - second < d * largest) grow with d for small d.
These are facts about random normal numbers, not about the model."""
import json

import numpy as np

rng = np.random.default_rng(0)
x = np.sort(rng.standard_normal((20_000_000, 10)).astype(np.float32), 1)
m1, m2, m3 = x[:, -1].astype(np.float64), x[:, -2].astype(np.float64), x[:, -3].astype(np.float64)
pos = m2 > 0
c = {}
for name, gap, scale in (("23", m2 - m3, m2), ("12", m1 - m2, m1)):
    ds = [0.005, 0.01, 0.02]
    rates = [np.mean(pos & (gap < d * scale)) / d for d in ds]
    c[name] = rates[0]
    print(f"c{name}: P(gap < d x value)/d for d = {ds}: {np.round(rates, 3)}")
one_pos = 11 / 1024 * 8 / 9
rows = json.load(open("bias_width_rows.json"))
print(f"\nerror = {one_pos:.4f} + {c['23']:.3f} * w23 + {c['12']:.3f} * w12   (no fitted constants)")
for kind in ("trained", "noise 0.5%", "noise 1.0%"):
    ks = [k for k in rows if rows[k]["kind"] == kind]
    err = np.array([1 - rows[k]["acc"] for k in ks])
    pred = np.array([one_pos + c["23"] * rows[k]["w23"] + c["12"] * rows[k]["w12"] for k in ks])
    rmse = np.sqrt(np.mean((pred - err) ** 2))
    print(f"{kind:11s}: typical error {100*err.mean():.2f} pts, predicted {100*pred.mean():.2f}; root-mean-square miss {100*rmse:.2f} pts; "
          f"constant guess misses by {100*err.std():.2f}; corr {np.corrcoef(pred, err)[0,1]:+.2f}")
    if kind == "trained":
        for k in ks:
            print(f"   {k:7s} error {100*(1-rows[k]['acc']):5.2f}  predicted {100*(one_pos + c['23']*rows[k]['w23'] + c['12']*rows[k]['w12']):5.2f}")
