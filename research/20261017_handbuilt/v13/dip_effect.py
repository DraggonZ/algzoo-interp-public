"""Same twin-run lists as overtake.py. Block the dip: copy neurons 0, 14, 15 from the twin (no overtaking) into the real
run at step L+2, then continue. Mean change of each final score (minus the mean score), by position relative to the old
largest r, and its slope on the old largest's value P (regression over lists, per offset)."""
import numpy as np
from algzoo.core import example
wi, wh, wo = (np.asarray(w, dtype=float) for w in example()); wi = wi.reshape(-1)
rng = np.random.default_rng(60); X = rng.standard_normal((800_000, 10)); B = len(X)
m1p = np.stack([np.max(np.c_[np.zeros((B, 1)), X[:, :t]], 1) for t in range(10)], 1); rec = X > m1p
rp = [np.flatnonzero(q) for q in rec]
r = np.array([q[-2] if len(q) >= 2 else -1 for q in rp]); L = np.array([q[-1] if len(q) else -1 for q in rp])
sec = X.argsort(1)[:, -2]
keep = (L - r >= 3) & (L <= 7) & (r >= 1) & (sec == r)
X = X[keep]; r = r[keep]; L = L[keep]; n = len(X); i = np.arange(n)
Xt = X.copy(); Xt[i, L] = -0.5
def run(X, patch=None):
    h = np.zeros((len(X), 16)); H = []
    for t in range(10):
        h = np.maximum(h @ wh.T + X[:, t:t + 1] * wi, 0)
        if patch is not None: h = patch(t, h)
        H.append(h)
    return np.stack(H, 1)
H, Ht = run(X), run(Xt)
for idx, name in (([0, 14, 15], "0, 14, 15"), ([10, 11, 12, 13], "10-13")):
    def patch(t, h, idx=idx):
        m = (L + 2) == t
        if m.any(): h = h.copy(); h[np.ix_(m, idx)] = Ht[np.flatnonzero(m), t][:, idx]
        return h
    S0 = H[:, 9] @ wo.T; S1 = run(X, patch)[:, 9] @ wo.T
    dS = (S1 - S1.mean(1, keepdims=True)) - (S0 - S0.mean(1, keepdims=True))
    P = X[i, r]
    print(f"--- block the change in {name} at L+2: score change by offset from the old largest r (mean | slope on P)")
    for off in range(-3, 6):
        p = r + off; ok = (p >= 0) & (p <= 9) & (p != L)
        y = dS[i[ok], p[ok]]; A = np.c_[P[ok], np.ones(ok.sum())]; co = np.linalg.lstsq(A, y, rcond=None)[0]
        print(f"  offset {off:+d}: mean {y.mean():6.1f} | slope on P {co[0]:6.1f}")
    y = dS[i, L]; co = np.linalg.lstsq(np.c_[P, X[i, L], np.ones(n)], y, rcond=None)[0]
    print(f"  at the current largest L: mean {y.mean():6.1f} | slope on P {co[0]:6.1f}, on x_L {co[1]:6.1f}")
