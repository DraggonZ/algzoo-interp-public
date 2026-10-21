"""Transfer: the decision-level model (probe_v2.py, canonical values 1.2 / 1.0 / 0.85, margin changes relative to the
clean margin; three quantities, saturating quadratic link with three fitted numbers) fitted on ARC's 200 copies at
s = 0.0025, applied unchanged to 150 copies (s = 0.0025, truth on 1M lists) of three other trained networks (seeds 106,
111, 115). The quantities need no role labels: they are the runner-up's margins on the same canonical lists. Also
refitted per network (10-fold cross-validated) to separate 'the quantities carry the response' from 'ARC's
coefficients carry over'."""
import numpy as np
from scipy.optimize import least_squares
from noiselib import net
from probe_v2 import feats

VALS = (1.2, 1.0, 0.85)
C = np.load("copies.npz")
def link(clean): return lambda p, z: 0.1 + (clean - 0.1) * np.exp(-((z ** 2) @ p))
def fit(f, z, y): return least_squares(lambda p: f(p, z) - y, np.ones(z.shape[1])).x
def rep(p, t): return f"RMSE {100 * np.sqrt(np.mean((p - t) ** 2)):5.2f}, correlation {np.corrcoef(p, t)[0, 1]:.3f}"
W = net("ARC"); fA = link(C["ARC_0.0025_clean"]); zA = feats(W, C["ARC_0.0025_dw"], VALS, True); pA = fit(fA, zA, C["ARC_0.0025_acc"])
print(f"ARC coefficients: {np.round(pA, 3)}; ARC's own quantities' SD across copies {np.round(zA.std(0), 3)}")
for s in ("106", "111", "115"):
    Wn = net(s); y = C[f"{s}_0.0025_acc"]; f = link(C[f"{s}_0.0025_clean"]); z = feats(Wn, C[f"{s}_0.0025_dw"], VALS, True)
    idx = np.random.default_rng(0).permutation(len(y)); pr = np.zeros(len(y))
    for fo in range(10):
        te = idx[fo::10]; tr = np.setdiff1d(idx, te); pr[te] = f(fit(f, z[tr], y[tr]), z[te])
    pa = f(pA, z)
    print(f"network {s}: clean {100 * C[f'{s}_0.0025_clean']:.2f}%, copies mean {100 * y.mean():.2f}, SD {100 * y.std():.2f}; quantities' SD {np.round(z.std(0), 3)}")
    print(f"   ARC's model unchanged: {rep(pa, y)}; mean predicted {100 * pa.mean():.2f}")
    print(f"   refitted on this network (CV): {rep(pr, y)}; coefficients {np.round(fit(f, z, y), 3)}")
