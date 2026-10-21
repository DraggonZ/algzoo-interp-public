"""What does the drive into the four hiding neurons depend on? Exact identity: pre_n(t) = c_n (ref_n(t) - x_t), with
c_n = -w_hi[n] and ref_n(t) = (W_hh h(t-1))_n / c_n. Split ref into the four's own recurrence (W_cc c(t-1)) and the
drive from the rest (store, detector, gates, memory). Regimes: r = position of the largest of x_0..x_{t-1} (floor 0;
r = -1 if none positive), K = t - r. Least squares per (neuron, r, K) of the drive on growing feature sets, no
intercept (the network has no biases, so everything is homogeneous in the numbers). Report residual RMS in
pre-activation units (median over regimes, weighted by steps) and the share of steps with residual above 0.2."""
import numpy as np
from common import WI, WH, C, lists, run
X = lists(2, 300_000); B = len(X); _, P, H = run(X)
Wcc = WH[np.ix_(C, C)]
Hp = np.concatenate([np.zeros_like(H[:, :1]), H[:, :-1]], 1)
drive = P[:, :, C] - Hp[:, :, C] @ Wcc.T - X[:, :, None] * WI[C]          # (B, 10, 4)
full = P[:, :, C] - X[:, :, None] * WI[C]                                   # c_n ref_n
# state before step t
M = np.zeros((B, 10)); Pp = np.zeros((B, 10)); R = np.full((B, 10), -1); m = np.zeros(B); p = np.zeros(B); r = np.full(B, -1)
for t in range(10):
    M[:, t], Pp[:, t], R[:, t] = m, p, r
    new = X[:, t] > m; p = np.where(new, m, p); m = np.where(new, X[:, t], m); r = np.where(new, t, r)
srt_kept = None


def feats(level, t, rr, idx):
    f = [Pp[idx, t], M[idx, t]]
    if level >= 1:   # numbers since the record (between r and t)
        for j in range(rr + 1, t):
            f += [X[idx, j], np.maximum(X[idx, j], 0)]
    if level >= 2:   # numbers before the record, other than the previous largest
        for j in range(0, max(rr, 0)):
            f += [np.maximum(X[idx, j], 0)]
    return np.stack(f, 1)


for target_name, T in (("drive from the rest", drive), ("whole reference (x c)", full)):
    print(f"== {target_name}")
    for level, lab in ((0, "previous largest, largest"), (1, "+ numbers since the record"), (2, "+ numbers before the record")):
        res = {n: [] for n in C}; cnt = []
        for t in range(1, 10):
            for rr in range(-1, t):
                idx = np.where(R[:, t] == rr)[0]
                if len(idx) < 200: continue
                A = feats(level, t, rr, idx)
                for i, n in enumerate(C):
                    y = T[idx, t, i]; co = np.linalg.lstsq(A, y, rcond=None)[0]; res[n].append(y - A @ co)
        out = []
        for n in C:
            e = np.concatenate(res[n]); out.append(f"{n}: rms {np.sqrt((e ** 2).mean()):5.2f}, >0.2 {np.mean(np.abs(e) > 0.2):4.2f}")
        print(f"  {lab:32s} " + " | ".join(out))
