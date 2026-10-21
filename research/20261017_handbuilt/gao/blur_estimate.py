"""ARC-angle proposal: add to the run-3 sampling-free estimate
(20261016_run3/scripts/reduce3_det.py: top three exact, the other seven at b = -0.46) the first-order effect of the seven
small numbers on the close-call bands. For each placement, one backward pass from the tie margins M23 and M12 gives
dM/dx at the seven background positions; with the small numbers' exchangeable covariance (variance 0.57 each, from the
order statistics of ten normals, no network runs), the margin is blurred, M ~ N(M0, s^2), and in the linear regime the
expected band width is E[max(-M, 0)] / slope = (mu Phi(mu/s) + s phi(mu/s)) / slope with mu = -M0. Deductive: no
sampling of the network. Developed on the six original networks (ARC's and zoo seeds 0-4)."""
import sys, itertools
import numpy as np
from scipy.special import ndtr
sys.path.insert(0, "/data/output/research/algzoo/20261016_run3/scripts")
import reduce3_det as R

rng = np.random.default_rng(0); S = np.sort(rng.standard_normal((400_000, 10)), 1)[:, :7]
S = np.take_along_axis(S, rng.permuted(np.tile(np.arange(7), (len(S), 1)), axis=1), 1)
COV = np.cov(S.T); VAR, CV = np.mean(np.diag(COV)), np.mean(COV[~np.eye(7, dtype=bool)])
phi = lambda z: np.exp(-z ** 2 / 2) / np.sqrt(2 * np.pi)


def back_margin(params, X, p2):
    """Margin of p2 over the best other at the end, and its gradient with respect to every input position."""
    w_hi, w_hh, w_oh = params; n = len(X); h = np.zeros((n, len(w_hi))); ons = []
    for t in range(10):
        z = h @ w_hh.T + X[:, t, None] * w_hi; on = (z > 0).astype(float); ons.append(on); h = z * on
    s = h @ w_oh.T; M, k = R.margin(s, p2)
    g = w_oh[p2] - w_oh[k]; dX = np.zeros((n, 10))
    for t in range(9, -1, -1):
        gz = g * ons[t]; dX[:, t] = gz @ w_hi; g = gz @ w_hh
    return M, dX


def blurred_width(M0, slope, sig, D):
    mu = -M0; z = mu / np.maximum(sig, 1e-12)
    ew = (mu * ndtr(z) + sig * phi(z)) / np.maximum(slope, 1e-12)
    return np.minimum(ew, D)


def estimate(params, blur=True):
    params = tuple(np.asarray(q, np.float64) for q in params)
    base, ops, det = R.estimate(params)
    if not blur:
        return base, ops, det
    T = np.array(list(itertools.permutations(range(10), 3))); n = len(T); ar = np.arange(n)
    run_cost = 10 * (len(params[0]) ** 2 + len(params[0])) + params[2].size
    def inputs(m1, m2, m3):
        X = np.full((n, 10), R.B); X[ar, T[:, 0]] = m1; X[ar, T[:, 1]] = m2; X[ar, T[:, 2]] = m3; return X
    bg = np.ones((n, 10), bool); bg[ar, T[:, 0]] = bg[ar, T[:, 1]] = bg[ar, T[:, 2]] = False
    W = {}
    for lab, X, Xe, D in (("23", inputs(R.L_MID + R.D12, R.L_MID, R.L_MID), inputs(R.L_MID + R.D12, R.L_MID, R.L_MID - 0.02), R.D23),
                          ("12", inputs(R.L_MID, R.L_MID, R.L_MID - R.D23), inputs(R.L_MID + 0.02, R.L_MID, R.L_MID - R.D23), R.D12)):
        M0, dX = back_margin(params, X, T[:, 1]); Me, _ = back_margin(params, Xe, T[:, 1]); ops += 3 * n * run_cost
        slope = (Me - M0) / 0.02
        gb = np.where(bg, dX, 0.0); sig = np.sqrt(np.maximum(VAR * (gb ** 2).sum(1) + CV * (gb.sum(1) ** 2 - (gb ** 2).sum(1)), 0))
        w_lin = blurred_width(M0, slope, sig, D)
        w_old = det["w" + lab]
        lin = (slope > 0) & (w_old < 0.02 + 1e-9)            # tie band within the linear regime (or no band)
        W[lab] = np.where(lin, w_lin, w_old)
    sc = np.maximum((R.M2[None, :] - R.B) / (R.L_MID - R.B), 0)
    Mm = det["Mlo"][:, None] + (det["Mhi"] - det["Mlo"])[:, None] * (R.M2[None, :] - R.L_LO) / (R.L_HI - R.L_LO)
    C = (Mm > 0).astype(float)
    f = 720 * R.PHI2[None, :] * C * (1 - ndtr(R.M2[None, :] + W["12"][:, None] * sc)) * ndtr(R.M2[None, :] - W["23"][:, None] * sc) ** 8 / 8
    return (f * R.W2[None, :]).sum(1).mean(), ops, W


if __name__ == "__main__":
    from algzoo.core import example, zoo
    truth = {"example": 0.9532, 0: 0.9101, 1: 0.8804, 2: 0.9485, 3: 0.9320, 4: 0.8898}
    print(f"small numbers: variance {VAR:.3f}, covariance {CV:.3f}")
    e0, e1 = [], []
    for name in ["example", 0, 1, 2, 3, 4]:
        p = example() if name == "example" else zoo(16, 10, name)
        a, _, _ = estimate(p, blur=False); b, ops, _ = estimate(p)
        e0.append(a - truth[name]); e1.append(b - truth[name])
        print(f"{name}: true {truth[name]:.4f}; run-3 estimate {a:.4f} ({100 * (a - truth[name]):+.2f}); with blur {b:.4f} ({100 * (b - truth[name]):+.2f}); cost {ops / 2880:.0f} runs' worth", flush=True)
    print(f"RMSE over six: run-3 {100 * np.sqrt(np.mean(np.square(e0))):.2f} points; with blur {100 * np.sqrt(np.mean(np.square(e1))):.2f}")
