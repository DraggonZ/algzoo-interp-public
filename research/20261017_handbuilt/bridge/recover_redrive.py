"""Question: is the store's timing signal what lets the trained hiding neurons recover after an early silencing?
Same measurement as robust_twin/trained_recover.py (a kept number within 5% below the largest, at least two steps after
the record, silences the four fully or partly; angle between their state and the state with that number replaced by
-1, 0-3 steps later), for the trained network and for the network whose hiding neurons read 15.88 x the largest so far
instead of the store (store_redrive.py). If the store re-drives the recovery, the angle should stay large.
With an argument: the same with each state's mean over the four removed, so the angle compares their turning
pattern, not their shared level (which dominates the full state)."""
import numpy as np
from algzoo.core import example
wi, wh, wo = (np.asarray(p, float) for p in example())
C = [1, 6, 7, 8]; K = 15.88
rng = np.random.default_rng(82); X = rng.standard_normal((300000, 10)); B = len(X)
def run(X, redrive):
    N = len(X); h = np.zeros((N, 16)); m1 = np.zeros(N); hs = []; ps = []
    for t in range(10):
        pre = h @ wh.T + X[:, t:t + 1] * wi
        if not redrive: pre[:, C] += np.outer(K * m1 - h[:, 2], wh[C, 2])
        h = np.maximum(pre, 0); m1 = np.maximum(m1, X[:, t]); hs.append(h); ps.append(pre)
    return np.stack(hs, 1), np.stack(ps, 1)
cm = np.maximum.accumulate(X, 1); prev = np.c_[np.zeros((B, 1)), cm[:, :-1]]
pm = np.maximum.accumulate(np.c_[np.full(B, -np.inf), X], 1)[:, :-1]; rec = X > pm
last = np.full(B, -1); age = np.zeros((B, 10), int)
for t in range(10): last = np.where(rec[:, t], t, last); age[:, t] = t - last
import sys
WAVE = len(sys.argv) > 1
print("angles of the four's pattern with their common part removed" if WAVE else "angles of the four's full state")
for name, rd in (("trained", True), ("store's timing removed from the hiding neurons", False)):
    H, Pre = run(X, rd); angs = {k: [] for k in range(4)}
    for t in range(3, 7):
        x = X[:, t]; m = (x < prev[:, t]) & (x > 0.95 * prev[:, t]) & (age[:, t] >= 2) & ((Pre[:, t][:, C] <= 0).sum(1) > 0)
        idx = np.flatnonzero(m)[:5000]
        Xb = X[idx].copy(); Xb[:, t] = -1.0; Hb, _ = run(Xb, rd)
        for k in range(4):
            if t + k > 9: continue
            a = H[idx, t + k][:, C]; b = Hb[:, t + k][:, C]
            if WAVE: a = a - a.mean(1, keepdims=True); b = b - b.mean(1, keepdims=True)
            cos = (a * b).sum(1) / np.maximum(np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1), 1e-9)
            angs[k] += list(np.degrees(np.arccos(np.clip(cos, -1, 1))))
    print(f"{name:48s} angle to the unsilenced state, 0-3 steps after: " + ", ".join(f"{np.median(angs[k]):5.1f}" for k in range(4)))
