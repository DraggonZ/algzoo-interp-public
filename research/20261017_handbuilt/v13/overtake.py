"""What does the trained network's memory do right after the largest is overtaken?
Lists: old largest P at r, a bigger number at L (r + 3 <= L <= 7), no other new largest, and P is the second-largest.
Twin run: the same list with x_L replaced by -0.5 (nothing overtaken). Report (1) how each neuron differs between the
runs at steps L .. L+3, and how that difference scales with P and with x_L; (2) copy one memory neuron (or a group) from
the twin into the real run at one step, keep running, and count how often the network still names P."""
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
P, xL = X[i, r], X[i, L]
ok0 = (H[:, 9] @ wo.T).argmax(1) == r
print(f"{n} lists; network names the old largest in {ok0.mean():.3f}")
mem = [0, 3, 10, 11, 12, 13, 14, 15]
np.set_printoptions(precision=2, suppress=True, linewidth=170)
print("difference real - twin, mean over lists, neurons 0-15, at steps L .. L+3:")
for d in range(4):
    m = L + d <= 9
    print(f"  L+{d}: {(H[i, np.minimum(L + d, 9)] - Ht[i, np.minimum(L + d, 9)])[m].mean(0)}")
print("at L+2, regress each memory neuron's difference on P and x_L (coefficients):")
D2 = (H[i, L + 2] - Ht[i, L + 2])
A = np.c_[P, xL, np.ones(n)]
for k in mem:
    co = np.linalg.lstsq(A, D2[:, k], rcond=None)[0]; print(f"  neuron {k:2d}: P {co[0]:6.2f}  x_L {co[1]:6.2f}")
print("copy from the twin into the real run at one step, then continue: share still naming the old largest")
for d in (1, 2, 3):
    row = []
    for name, idx in (("all memory", mem), ("0", [0]), ("3", [3]), ("15", [15]), ("0,3,15", [0, 3, 15]), ("10-14", [10, 11, 12, 13, 14]),
                      ("cancellers", [1, 6, 7, 8]), ("store 2", [2])):
        def patch(t, h, idx=idx, d=d):
            m = (L + d) == t
            if m.any(): h = h.copy(); h[np.ix_(m, idx)] = Ht[np.flatnonzero(m), t][:, idx]
            return h
        ok = (run(X, patch)[:, 9] @ wo.T).argmax(1) == r
        row.append(f"{name} {ok.mean():.2f}")
    print(f"  at L+{d}: " + "  ".join(row))

# Before the overtaking (step L-1, the old largest still current): does each memory neuron hold a wave of the old
# largest's value P, or of its jump P - P_prev? k = steps since P arrived; angle 40 or 45 degrees per step.
recpos = [np.flatnonzero(X[j] > np.r_[0, np.maximum.accumulate(np.maximum(X[j], 0))[:-1]]) for j in range(n)]
Pp = np.array([X[j, q[q < rr][-1]] if (q < rr).any() else 0.0 for j, (q, rr) in enumerate(zip(recpos, r))])
k = (L - 1) - r; hL = H[i, L - 1]
print("\nbefore the overtaking (step L-1): R2 of each memory neuron from a wave of the value vs a wave of the jump")
for deg in (40, 45):
    th = np.radians(deg) * k; out = []
    for kk in mem:
        y = hL[:, kk]
        def r2(A):
            A = np.c_[A, np.ones(n)]; co = np.linalg.lstsq(A, y, rcond=None)[0]; return 1 - ((y - A @ co) ** 2).mean() / y.var()
        rv = r2(np.c_[P * np.cos(th), P * np.sin(th), P]); rj = r2(np.c_[(P - Pp) * np.cos(th), (P - Pp) * np.sin(th), P])
        out.append(f"{kk}: value {rv:.2f} jump {rj:.2f}")
    print(f"  {deg} deg/step  " + " | ".join(out))
