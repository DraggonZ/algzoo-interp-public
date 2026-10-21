"""Decision-level explanation quantities read from each copy's weights by running it on fixed canonical lists (no
sampling): the three largest at every arrangement of positions (720), values largest 1.2, runner-up 1.0, third 0.85,
the other seven at -0.46 (the truncated mean used by run 3's reduction). Per arrangement, the runner-up's margins over
the largest and over the third (scores). Quantities per copy: the change from the clean network of
 - the largest's leak: margin over the largest, mean and RMS change, separately for runner-up before / after the largest;
 - the third's margin: mean and RMS change.
Then a small quadratic model of accuracy in these (10-fold cross-validated), its transfer to s = 0.001 and 0.005, and to
the shared held-out copies (truth/noisy_truth.npz, 2M lists, s = 0.001-0.01)."""
import itertools, numpy as np, torch as th
from noiselib import net, flat, unflat, DEV, DT
from screen import cv, fit_full, design

def canon():
    rows, arr = [], []
    for pL, pS, pT in itertools.permutations(range(10), 3):
        x = np.full(10, -0.46); x[pL], x[pS], x[pT] = 1.2, 1.0, 0.85; rows.append(x); arr.append((pL, pS, pT))
    return th.as_tensor(np.array(rows), device=DEV, dtype=DT), np.array(arr)
XC, ARR = canon()

def margins(W, dws):
    wi, wh, wo = unflat(flat(W)[None] + dws, W)
    h = th.zeros(len(dws), len(XC), 16, device=DEV, dtype=DT)
    for t in range(10): h = th.relu(th.einsum("cbj,cij->cbi", h, wh) + XC[None, :, t:t + 1] * wi[:, None, :])
    s = th.einsum("cbj,cij->cbi", h, wo); i = th.arange(len(XC), device=DEV)
    a = th.as_tensor(ARR, device=DEV)
    sS = s[:, i, a[:, 1]]; return (sS - s[:, i, a[:, 0]]).cpu().numpy(), (sS - s[:, i, a[:, 2]]).cpu().numpy()

def features(W, dws):
    mL0, mT0 = margins(W, th.zeros(1, 432, device=DEV, dtype=DT)); mL, mT = margins(W, dws)
    dL, dT = mL - mL0, mT - mT0; before = ARR[:, 1] < ARR[:, 0]
    f = {"largest's leak, runner-up before: mean": dL[:, before].mean(1), "largest's leak, runner-up before: RMS": np.sqrt((dL[:, before] ** 2).mean(1)),
         "largest's leak, runner-up after: mean": dL[:, ~before].mean(1), "largest's leak, runner-up after: RMS": np.sqrt((dL[:, ~before] ** 2).mean(1)),
         "third's margin: mean": dT.mean(1), "third's margin: RMS": np.sqrt((dT ** 2).mean(1))}
    return f, (mL0, mT0)

if __name__ == "__main__":
    W = net("ARC"); C = np.load("copies.npz"); T = np.load("../truth/noisy_truth.npz")
    f0, (mL0, mT0) = features(W, th.zeros(1, 432, device=DEV, dtype=DT))
    print(f"clean canonical margins: over the largest median {np.median(mL0):.2f} (runner-up wrong on {np.mean(mL0 < 0):.3f} of arrangements), over the third median {np.median(mT0):.2f} (wrong on {np.mean(mT0 < 0):.3f})")
    F = {s: features(W, th.as_tensor(C[f"ARC_{s}_dw"], device=DEV))[0] for s in ("0.0025", "0.001", "0.005")}
    H = {s: features(W, th.as_tensor(T[f"dw_{s}"], device=DEV))[0] for s in ("0.001", "0.0025", "0.005", "0.01")}
    y = C["ARC_0.0025_acc"]; names = list(F["0.0025"])
    def rep(p, t): return f"RMSE {100 * np.sqrt(np.mean((p - t) ** 2)):.2f} points, correlation {np.corrcoef(p, t)[0, 1]:.3f}"
    for k in names:
        print(f"  {k:42s} SD across copies {F['0.0025'][k].std():6.2f}; alone (quadratic, CV): {rep(cv(F['0.0025'][k][:, None], y), y)}")
    for sel in (["largest's leak, runner-up before: RMS", "largest's leak, runner-up after: RMS", "third's margin: RMS"], names):
        Z = np.stack([F["0.0025"][k] for k in sel], 1); f, npar = fit_full(Z, y)
        print(f"\nmodel on {len(sel)} quantities ({npar} fitted numbers), 200 copies at s = 0.0025, cross-validated: {rep(cv(Z, y), y)}")
        for s in ("0.001", "0.005"):
            print(f"   my copies s = {s} (no refit): {rep(f(np.stack([F[s][k] for k in sel], 1)), C[f'ARC_{s}_acc'])}")
        for s in H:
            print(f"   shared held-out copies s = {s} (no refit): {rep(f(np.stack([H[s][k] for k in sel], 1)), T[f'stats_{s}'][:, 0])}; constant guess {100 * T[f'stats_{s}'][:, 0].std():.2f}")
