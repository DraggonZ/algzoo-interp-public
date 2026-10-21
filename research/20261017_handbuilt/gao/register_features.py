"""Can the register's own rotation rates give the cancelling neurons' direction more exactly? The rotation rates and
decays come from the weights: eigenvalues of the recurrent matrix with the most common on/off pattern fixed (steps 3-9).
Features for neurons 1, 6, 7, 8: x, m1, m2, and for each complex eigenvalue pair (angle th, size rho) the waves
m1 rho^age cos(age th), m1 rho^age sin(age th), and the same with the record's jump d as height. Least squares on
pre-activations (as matched_size.py). Reports angle to the true pattern, size ratio and accuracy, against our account."""
from collections import Counter
import numpy as np
import simlib
from simlib import task_vars
from sim_rich import record_vars
from matched_size import X, y, H, Xf, PREf, account_features
from simlib import run

Hf = simlib.run(Xf[:20_000])[2][:, 3:] > 0
pat = Counter(map(bytes, np.packbits(Hf.reshape(-1, 16), axis=1))).most_common(1)[0][0]
mask = np.unpackbits(np.frombuffer(pat, np.uint8))[:16].astype(float)
ev = np.linalg.eigvals(simlib.WH * mask[:, None])
pairs = sorted({(round(abs(np.angle(e)), 4), round(abs(e), 4)) for e in ev if np.imag(e) > 1e-6 or (np.real(e) < -0.3 and abs(np.imag(e)) < 1e-6)})
print("rotation pairs (angle deg, size per step):", [(round(np.degrees(a), 1), r) for a, r in pairs])
ns = [1, 6, 7, 8]


def feats(Xs):
    V = task_vars(Xs); x, m1, m2, age = V["x"], V["m1"], V["m2"], V["age"]; live = age >= 0
    prev, _, _ = record_vars(Xs); d = np.where(live, m1 - prev, 0)
    F = [x, m1, m2]
    for th, rho in pairs:
        for hgt in (m1, d):
            amp = np.where(live, hgt * rho ** np.maximum(age, 0), 0)
            F += [amp * np.cos(age * th), amp * np.sin(age * th)]
    return np.stack(F, -1)


def evaluate(lab, fn):
    sims = {}
    for n in ns:
        Ff = fn(Xf, n); co = np.linalg.lstsq(Ff.reshape(-1, Ff.shape[-1]), PREf[:, :, n].reshape(-1), rcond=None)[0]
        sims[n] = np.maximum(fn(X, n) @ co, 0)
    F = np.stack([sims[n] for n in ns], -1); V = H[:, :, ns]
    nf = np.linalg.norm(F, axis=-1); nv = np.linalg.norm(V, axis=-1); act = nv > 1e-6
    cos = (F * V).sum(-1)[act] / np.maximum(nf[act] * nv[act], 1e-9)
    acc = (run(X, sims)[0].argmax(1) == y).mean()
    print(f"{lab:38s}: {F.shape[-1] if False else ''}angle median {np.degrees(np.arccos(np.clip(np.median(cos), -1, 1))):.1f} deg; size ratio median {np.median(nf[act] / nv[act]):.3f}; accuracy {acc:.3f}", flush=True)


evaluate("our account (5 quantities)", account_features)
FX, FS = feats(Xf), feats(X)
evaluate(f"register waves ({FX.shape[-1]} quantities)", lambda Xs, n: FX if Xs is Xf else FS)
from sim_across import feats as menu  # the role-free 30-quantity menu (largest's wave at five fixed angles, no decay)
MX, MS = menu(Xf), menu(X)
evaluate(f"general menu ({MX.shape[-1]} quantities)", lambda Xs, n: MX if Xs is Xf else MS)
BX, BS = np.concatenate([FX, MX], -1), np.concatenate([FS, MS], -1)
evaluate(f"register waves + menu ({BX.shape[-1]})", lambda Xs, n: BX if Xs is Xf else BS)
