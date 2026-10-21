"""A shaped link from the three decision-level quantities of probe_margins.py (RMS change of the runner-up's margin over
the largest, runner-up before / after it, and over the third, on the 720 canonical lists) to accuracy, so that it can
extrapolate across noise sizes. Mechanism: a margin shift of size d flips the lists whose margin is below d, so for
small shifts the loss grows linearly in the shift's size; it saturates at chance (10%).
 linear:      acc = clean - sum_c k_c RMS_c                            (3 fitted numbers)
 saturating:  acc = 0.1 + (clean - 0.1) exp(-sum_c k_c RMS_c)           (3 fitted numbers)
 sat. + sq.:  as saturating, exponent sum_c k_c RMS_c + q (sum RMS)^2   (4 fitted numbers)
 sat. quadratic: acc = 0.1 + (clean - 0.1) exp(-z^T Q z), Q diagonal (3) or full (6): at the trained optimum gains and
              losses of lists cancel to first order, so the loss starts quadratic in the shifts
Fitted on the 200 copies at s = 0.0025 (10-fold cross-validated there), applied unchanged to s = 0.001-0.01 (my copies
and the shared held-out copies, truth on 2M lists). Also: mistakes naming the largest, from the same quantities."""
import numpy as np, torch as th
from scipy.optimize import least_squares
from noiselib import net, DEV, DT
from probe_margins import features

SEL = ["largest's leak, runner-up before: RMS", "largest's leak, runner-up after: RMS", "third's margin: RMS"]
W = net("ARC"); C = np.load("copies.npz"); T = np.load("../truth/noisy_truth.npz")
clean = T["arc"][0]
def Z(dw): f = features(W, th.as_tensor(dw, device=DEV))[0]; return np.stack([f[k] for k in SEL], 1)
train = (Z(C["ARC_0.0025_dw"]), C["ARC_0.0025_acc"])
tests = {f"my s = {s}": (Z(C[f"ARC_{s}_dw"]), C[f"ARC_{s}_acc"]) for s in ("0.001", "0.005")}
tests.update({f"held-out s = {s}": (Z(T[f"dw_{s}"]), T[f"stats_{s}"][:, 0]) for s in ("0.001", "0.0025", "0.005", "0.01")})
models = {
    "linear": (3, lambda p, z: clean - z @ p[:3]),
    "saturating": (3, lambda p, z: 0.1 + (clean - 0.1) * np.exp(-(z @ p[:3]))),
    "saturating + square": (4, lambda p, z: 0.1 + (clean - 0.1) * np.exp(-(z @ p[:3]) - p[3] * z.sum(1) ** 2)),
    "saturating quadratic, diagonal": (3, lambda p, z: 0.1 + (clean - 0.1) * np.exp(-((z ** 2) @ p[:3]))),
    "saturating quadratic, full": (6, lambda p, z: 0.1 + (clean - 0.1) * np.exp(-((z ** 2) @ p[:3]) - p[3] * z[:, 0] * z[:, 1] - p[4] * z[:, 0] * z[:, 2] - p[5] * z[:, 1] * z[:, 2])),
}
def fit(m, z, y):
    k, f = models[m]; return least_squares(lambda p: f(p, z) - y, np.full(k, 0.0003)).x
def rep(p, t): return f"RMSE {100 * np.sqrt(np.mean((p - t) ** 2)):5.2f}, correlation {np.corrcoef(p, t)[0, 1]:.3f}"
z, y = train; idx = np.random.default_rng(0).permutation(len(y))
for m, (k, f) in models.items():
    pr = np.zeros(len(y))
    for fo in range(10):
        te = idx[fo::10]; tr = np.setdiff1d(idx, te); pr[te] = f(fit(m, z[tr], y[tr]), z[te])
    p = fit(m, z, y)
    print(f"{m} ({k} fitted numbers; coefficients {np.round(p, 4)}): s = 0.0025 cross-validated {rep(pr, y)}")
    for tn, (zt, yt) in tests.items():
        print(f"    {tn:22s} {rep(f(p, zt), yt)}; constant guess {100 * yt.std():5.2f}; mean true {100 * yt.mean():.2f} predicted {100 * f(p, zt).mean():.2f}")
# mistakes naming the largest (share of all lists): from the largest's-leak quantities, linear in RMS, fitted on held-out s = 0.0025
print("\nShare of lists answered with the largest (held-out copies): fitted on s = 0.0025, applied to the others")
lg = {s: (1 - T[f"stats_{s}"][:, 0]) * T[f"stats_{s}"][:, 1] for s in ("0.001", "0.0025", "0.005", "0.01")}
zs = {s: Z(T[f"dw_{s}"]) for s in lg}
c0 = (1 - clean) * T["arc"][1]
for cols, lab in (([0, 1], "largest's leak only"), ([0, 1, 2], "all three")):
    A = zs["0.0025"][:, cols]; coef = np.linalg.lstsq(A, lg["0.0025"] - c0, rcond=None)[0]
    for s in lg:
        p = c0 + zs[s][:, cols] @ coef
        print(f"  {lab:22s} s = {s:6s}: true mean {100 * lg[s].mean():5.2f}% of lists, predicted {100 * p.mean():5.2f}; per copy {rep(p, lg[s])}")
