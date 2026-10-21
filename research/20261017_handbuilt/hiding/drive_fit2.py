"""Which source leaves the residual of drive_fit.py? Split the drive into the four by source (store, detector, keep
gate, late gate, memory) and fit each per (neuron, r, K) on: (a) previous largest P and largest M; (b) + numbers since
the record; (c) + P times an indicator of where P arrived, and the numbers since times an indicator of whether the gate
kept them (x_j at least 0.75 of the second-largest at j). Residual RMS in pre-activation units, steps 1-9."""
import numpy as np
from common import WI, WH, C, GROUPS, lists, run
X = lists(2, 300_000); B = len(X); _, P, H = run(X)
Hp = np.concatenate([np.zeros_like(H[:, :1]), H[:, :-1]], 1)
src = {g: Hp[:, :, idx] @ WH[np.ix_(C, idx)].T for g, idx in GROUPS.items() if g != "hiding"}
M = np.zeros((B, 10)); Pp = np.zeros((B, 10)); R = np.full((B, 10), -1); RP = np.full((B, 10), -1); M2 = np.zeros((B, 10))
m = np.zeros(B); p = np.zeros(B); r = np.full(B, -1); rp = np.full(B, -1); m2 = np.zeros(B)
KEPT = np.zeros((B, 10))
for t in range(10):
    M[:, t], Pp[:, t], R[:, t], RP[:, t], M2[:, t] = m, p, r, rp, m2
    x = X[:, t]; new = x > m
    KEPT[:, t] = (x >= 0.75 * m2) & (x > 0)
    rp = np.where(new, r, rp); m2 = np.where(new, m, np.maximum(m2, x))
    p = np.where(new, m, p); m = np.where(new, x, m); r = np.where(new, t, r)


def feats(level, t, rr, idx):
    f = [Pp[idx, t], M[idx, t]]
    if level >= 1:
        for j in range(rr + 1, t): f += [X[idx, j], np.maximum(X[idx, j], 0)]
    if level >= 2:
        for j in range(-1, max(rr, 0)): f += [Pp[idx, t] * (RP[idx, t] == j)]
        for j in range(rr + 1, t): f += [X[idx, j] * KEPT[idx, j]]
    return np.stack(f, 1)


for g, T in src.items():
    out = []
    for level in (0, 1, 2):
        res = []
        for t in range(1, 10):
            for rr in range(-1, t):
                idx = np.where(R[:, t] == rr)[0]
                if len(idx) < 300: continue
                A = feats(level, t, rr, idx); y = T[idx, t]
                co = np.linalg.lstsq(A, y, rcond=None)[0]; res.append(y - A @ co)
        e = np.concatenate(res); out.append("(" + ", ".join(f"{np.sqrt((e[:, i] ** 2).mean()):4.2f}" for i in range(4)) + ")")
    print(f"{g:10s} residual rms for neurons 1, 6, 7, 8 at levels a / b / c: " + " / ".join(out))
