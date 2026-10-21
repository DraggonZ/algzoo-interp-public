"""Review 2 check: what simple corrections of existing estimators would give on the 96-copy benchmark.
Uses saved outputs only (additive_pilot.npz, 2^19 sampled inputs per copy for the surrogates)."""
import numpy as np
d = np.load('/data/output/research/algzoo/20261016_run2/reports/additive_pilot.npz', allow_pickle=True)
rows, names = d['rows'], d['names']
cols = list(d['cols'])
sig = np.array([float(str(n).split('_')[0][4:]) for n in names])
t = rows[:, 0]
base_acc = 0.9531
def rep(nm, e):
    out = [f"{nm:28s} all RMSE {np.sqrt(np.mean((e-t)**2)):.4f}"]
    for s in (0.005, 0.01, 0.02):
        k = sig == s
        out.append(f"σ={s}: {np.sqrt(np.mean((e[k]-t[k])**2)):.4f}")
    print(" | ".join(out))
for s in (0.005, 0.01, 0.02):
    k = sig == s
    print(f"σ={s}: n={k.sum()} true acc mean {t[k].mean():.3f} SD {t[k].std():.3f} min {t[k].min():.3f}; sampling RMSE@100 {np.sqrt(np.mean(t[k]*(1-t[k])/100)):.4f}")
print(f"all: sampling RMSE@100 {np.sqrt(np.mean(t*(1-t)/100)):.4f}  @1000 {np.sqrt(np.mean(t*(1-t)/1000)):.4f}")
rep("constant = base accuracy", np.full_like(t, base_acc))
for c in ("Z2", "Z3", "mean_bg", "additive", "additive_meanref"):
    rep(c, rows[:, cols.index(c)])
# oracle affine calibration (fit to truth: NOT allowed, only to show information content)
for c in ("Z2", "Z3", "additive", "additive_meanref"):
    e = rows[:, cols.index(c)]
    A = np.c_[np.ones_like(e), e]
    coef = np.linalg.lstsq(A, t, rcond=None)[0]
    rep(f"{c} oracle affine (not allowed)", A @ coef)
    print("   corr", np.corrcoef(e, t)[0, 1].round(4), "coef", coef.round(3))
# loss-ratio rule: true loss ≈ k * surrogate loss, k fixed from the base model? (needs base surrogate values)
