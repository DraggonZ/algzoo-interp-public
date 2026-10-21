"""Variants of the decision-level quantities (probe_margins.py): margins on canonical lists at near-tie values as well
(largest 1.05, runner-up 1.0, third 0.95), and each arrangement's margin change taken relative to its clean margin
(a shift matters in proportion to how close the arrangement is to flipping). Same saturating quadratic link (one
fitted number per quantity), fitted on the 200 copies at s = 0.0025, applied to the held-out shared copies."""
import itertools, numpy as np, torch as th
from scipy.optimize import least_squares
from noiselib import net, flat, unflat, DEV, DT

def canon(vals):
    rows, arr = [], []
    for pL, pS, pT in itertools.permutations(range(10), 3):
        x = np.full(10, -0.46); x[pL], x[pS], x[pT] = vals; rows.append(x); arr.append((pL, pS, pT))
    return th.as_tensor(np.array(rows), device=DEV, dtype=DT), np.array(arr)

def margins(W, dws, XC, ARR):
    wi, wh, wo = unflat(flat(W)[None] + dws, W)
    h = th.zeros(len(dws), len(XC), 16, device=DEV, dtype=DT)
    for t in range(10): h = th.relu(th.einsum("cbj,cij->cbi", h, wh) + XC[None, :, t:t + 1] * wi[:, None, :])
    s = th.einsum("cbj,cij->cbi", h, wo); i = th.arange(len(XC), device=DEV); a = th.as_tensor(ARR, device=DEV)
    sS = s[:, i, a[:, 1]]; return (sS - s[:, i, a[:, 0]]).cpu().numpy(), (sS - s[:, i, a[:, 2]]).cpu().numpy()

def feats(W, dw, vals, rel):
    XC, ARR = canon(vals); dw = th.as_tensor(dw, device=DEV)
    (mL0, mT0), (mL, mT) = margins(W, th.zeros(1, 432, device=DEV, dtype=DT), XC, ARR), margins(W, dw, XC, ARR)
    dL, dT = mL - mL0, mT - mT0
    if rel: dL, dT = dL / np.maximum(np.abs(mL0), 1.0), dT / np.maximum(np.abs(mT0), 1.0)
    b = ARR[:, 1] < ARR[:, 0]
    return np.stack([np.sqrt((dL[:, b] ** 2).mean(1)), np.sqrt((dL[:, ~b] ** 2).mean(1)), np.sqrt((dT ** 2).mean(1))], 1)

if __name__ == "__main__":
    W = net("ARC"); C = np.load("copies.npz"); T = np.load("../truth/noisy_truth.npz"); clean = T["arc"][0]
    f = lambda p, z: 0.1 + (clean - 0.1) * np.exp(-((z ** 2) @ p))
    def rep(p, t): return f"RMSE {100 * np.sqrt(np.mean((p - t) ** 2)):5.2f}, corr {np.corrcoef(p, t)[0, 1]:.3f}"
    y = C["ARC_0.0025_acc"]; idx = np.random.default_rng(0).permutation(len(y))
    for vals in ((1.2, 1.0, 0.85), (1.05, 1.0, 0.95)):
        for rel in (False, True):
            z = feats(W, C["ARC_0.0025_dw"], vals, rel)
            fit = lambda zz, yy: least_squares(lambda p: f(p, zz) - yy, np.full(zz.shape[1], 1e-3 if not rel else 1.0)).x
            pr = np.zeros(len(y))
            for fo in range(10):
                te = idx[fo::10]; tr = np.setdiff1d(idx, te); pr[te] = f(fit(z[tr], y[tr]), z[te])
            p = fit(z, y)
            line = f"values {vals}, {'relative to clean margin' if rel else 'absolute'}: CV s=0.0025 {rep(pr, y)} |"
            for s in ("0.001", "0.0025", "0.005", "0.01"):
                line += f" held-out {s}: {rep(f(p, feats(W, T[f'dw_{s}'], vals, rel)), T[f'stats_{s}'][:, 0])} |"
            print(line)
