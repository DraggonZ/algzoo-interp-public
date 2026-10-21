"""Ceilings: if some parts of a hiding neuron's input were known exactly, how small could the formula's error get?
(i) store and detector parts exact (ARC's values); (ii) store and detector from their own recursion (a rule with the
pair's 4 recurrent and 2 input weights, nothing from other neurons); (iii) plus the other hiding neurons' previous
values exact. Remaining parts fitted with the account's features."""
import numpy as np
from harness import *  # noqa
sys.path.insert(0, "/data/output/research/algzoo/20261017_handbuilt/gao")
from sim_groups import account_features

WI, WH = simlib.WI, simlib.WH


def sd_rule(X):
    """Store (2) and detector (4) from their own 2x2 recursion; returns values after each step, (B, 10, 2)."""
    B = len(X); s = np.zeros(B); d = np.zeros(B); out = np.zeros((B, 10, 2))
    for t in range(10):
        s, d = (np.maximum(WH[2, 2] * s + WH[2, 4] * d + WI[2] * X[:, t], 0),
                np.maximum(WH[4, 2] * s + WH[4, 4] * d + WI[4] * X[:, t], 0))
        out[:, t, 0], out[:, t, 1] = s, d
    return out


for lab, Hset in [("ARC", None)]:
    pass
Hprev_f = np.concatenate([np.zeros((len(Xf), 1, 16)), Hf[:, :-1]], 1)
Hprev_s = np.concatenate([np.zeros((len(Xs), 1, 16)), Hs[:, :-1]], 1)
SDf = np.concatenate([np.zeros((len(Xf), 1, 2)), sd_rule(Xf)[:, :-1]], 1)
SDs = np.concatenate([np.zeros((len(Xs), 1, 2)), sd_rule(Xs)[:, :-1]], 1)
print(f"store/detector rule vs ARC: rms error store {np.sqrt(((SDs[..., 0] - Hprev_s[..., 2]) ** 2).mean()):.3f} "
      f"(ARC rms {np.sqrt((Hprev_s[..., 2] ** 2).mean()):.2f}), detector {np.sqrt(((SDs[..., 1] - Hprev_s[..., 4]) ** 2).mean()):.3f}")


def known_plus_features(n, known_f, known_s):
    """Fit pre - known part on account features; report rms after ReLU and replaced accuracy."""
    Ff, Fs = account_features(Xf, n), account_features(Xs, n)
    co = np.linalg.lstsq(Ff.reshape(-1, Ff.shape[-1]), (Pf[:, :, n] - known_f).reshape(-1), rcond=None)[0]
    sim = np.maximum(Fs @ co + known_s, 0)
    rms = np.sqrt(((sim - Hs[:, :, n]) ** 2).mean())
    acc = score(Xs, Ls, run(Xs, {n: sim})[0])["acc"]
    return rms, acc


for n in HID:
    others = [k for k in HID if k != n]
    r = {}
    kf = Hprev_f[..., [2, 4]] @ WH[n, [2, 4]]; ks = Hprev_s[..., [2, 4]] @ WH[n, [2, 4]]
    r["(i) store+det exact"] = known_plus_features(n, kf, ks)
    kf2 = SDf @ WH[n, [2, 4]]; ks2 = SDs @ WH[n, [2, 4]]
    r["(ii) store+det by their rule"] = known_plus_features(n, kf2, ks2)
    kf3 = kf2 + Hprev_f[..., HID] @ WH[n, HID]; ks3 = ks2 + Hprev_s[..., HID] @ WH[n, HID]
    r["(iii) + hiding previous exact"] = known_plus_features(n, kf3, ks3)
    kf4 = kf3 + Hprev_f[..., [5, 9]] @ WH[n, [5, 9]]; ks4 = ks3 + Hprev_s[..., [5, 9]] @ WH[n, [5, 9]]
    r["(iv) + gates exact"] = known_plus_features(n, kf4, ks4)
    print(f"neuron {n} (account {BASE_RMS[n]:.3f}): " + "; ".join(f"{k}: rms {v[0]:.3f}, acc {v[1]:.3f}" for k, v in r.items()))
