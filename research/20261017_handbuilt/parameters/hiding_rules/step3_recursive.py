"""Hiding-group rule that runs on itself: each hiding neuron's input = a formula of the list (features) + a linear
combination of the four hiding neurons' RULE values at the previous step. Coefficients by least squares with ARC's
previous hiding values as regressors (teacher forcing, fit set); then the rule runs recursively on the score set (no
ARC values used). Feature sets:
  A: the account's 5 features (x, m1, m2, m1 x timer wave at the neuron's angle)
  B: A + store/detector from their own 2x2 rule (previous step) + kept numbers by age (9 lags) + m1 waves at the
     memory's modes (43, 83, 126 degrees with their decays, and 180)
Reported per neuron, and with all four replaced together (the group's rule)."""
import numpy as np
from harness import *  # noqa
sys.path.insert(0, "/data/output/research/algzoo/20261017_handbuilt/gao")
from sim_groups import account_features

WI, WH = simlib.WI, simlib.WH
MODES = [(43, 0.94), (83, 0.94), (126, 0.82), (180, 0.90)]   # memory+hiding block eigenvalues (block_rank.log)


def sd_rule(X):
    B = len(X); s = np.zeros(B); d = np.zeros(B); out = np.zeros((B, 10, 2))
    for t in range(10):
        s, d = (np.maximum(WH[2, 2] * s + WH[2, 4] * d + WI[2] * X[:, t], 0),
                np.maximum(WH[4, 2] * s + WH[4, 4] * d + WI[4] * X[:, t], 0))
        out[:, t, 0], out[:, t, 1] = s, d
    return np.concatenate([np.zeros((B, 1, 2)), out[:, :-1]], 1)   # previous step's values


def featsB(X, n):
    V = task_vars(X); x, m1, m2, age = V["x"], V["m1"], V["m2"], V["age"]; live = age >= 0
    kept = (x > 0) & (x >= 0.75 * m2); kv = np.where(kept, x, 0)
    F = [account_features(X, n)[..., k] for k in range(5)]
    F += [sd_rule(X)[..., 0], sd_rule(X)[..., 1]]
    F += [lag(kv, a) for a in range(1, 10)]
    for deg, rho in MODES:
        amp = np.where(live, m1 * rho ** np.maximum(age, 0), 0); th = np.radians(deg)
        F += [amp * np.cos(age * th)] + ([amp * np.sin(age * th)] if deg != 180 else [])
    return np.stack(F, -1)


def fit_group(feat_fn, label):
    Hprev_f = np.concatenate([np.zeros((len(Xf), 1, 16)), Hf[:, :-1]], 1)[..., HID]
    co = {}
    for n in HID:
        F = np.concatenate([feat_fn(Xf, n), Hprev_f], -1)
        co[n] = np.linalg.lstsq(F.reshape(-1, F.shape[-1]), Pf[:, :, n].reshape(-1), rcond=None)[0]
    Fs = {n: feat_fn(Xs, n) for n in HID}
    # teacher-forced (ARC's previous hiding values) and recursive (the rule's own previous values)
    Hprev_s = np.concatenate([np.zeros((len(Xs), 1, 16)), Hs[:, :-1]], 1)[..., HID]
    tf = {n: np.maximum(np.concatenate([Fs[n], Hprev_s], -1) @ co[n], 0) for n in HID}
    rec = np.zeros((len(Xs), 10, 4)); prev = np.zeros((len(Xs), 4))
    for t in range(10):
        for i, n in enumerate(HID):
            rec[:, t, i] = np.maximum(np.concatenate([Fs[n][:, t], prev], -1) @ co[n], 0)
        prev = rec[:, t]
    print(f"\n{label} (+4 hiding feedback terms; {Fs[1].shape[-1] + 4} constants per neuron)")
    for i, n in enumerate(HID):
        r_tf = np.sqrt(((tf[n] - Hs[:, :, n]) ** 2).mean()); r_rec = np.sqrt(((rec[:, :, i] - Hs[:, :, n]) ** 2).mean())
        sc = score(Xs, Ls, run(Xs, {n: rec[:, :, i]})[0])
        print(f"  neuron {n}: rms teacher-forced {r_tf:.3f}, recursive {r_rec:.3f} ({BASE_RMS[n] / r_rec:.2f}x the account); "
              f"replaced alone: accuracy {sc['acc']:.4f}, KL {sc['kl']:.3f}", flush=True)
    sc = score(Xs, Ls, run(Xs, {n: rec[:, :, i] for i, n in enumerate(HID)})[0])
    print(f"  all four replaced by the group rule: accuracy {sc['acc']:.4f}, KL {sc['kl']:.3f}")
    return co, rec


if __name__ == "__main__":
    fit_group(lambda X, n: account_features(X, n), "A: account features")
    fit_group(featsB, "B: richer features")
