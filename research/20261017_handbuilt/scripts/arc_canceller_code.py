"""What do the trained network's four cancelling neurons (1, 6, 7, 8) encode? At steps 2-9, where a canceller is on,
fit it (no intercept: the network has no biases) as a mix of: the input x_t, the largest before t (m1), the
second-largest before t (m2), and the current largest's timer wave m1*cos(k*th), m1*sin(k*th), where k is the number of
steps since that largest arrived. th is chosen by grid search. Then drop each quantity to see how much it matters."""
import numpy as np
from algzoo.core import example
wi, wh, wo = (np.asarray(w, dtype=float) for w in example()); wi = wi.reshape(-1)
X = np.random.default_rng(41).standard_normal((200_000, 10)); B = len(X)
H = []; h = np.zeros((B, 16))
for t in range(10): h = np.maximum(h @ wh.T + X[:, t:t + 1] * wi, 0); H.append(h)
H = np.stack(H, 1)
srt = [np.sort(np.c_[np.zeros((B, 2)), X[:, :t]], 1) for t in range(10)]
m1 = np.stack([s[:, -1] for s in srt], 1); m2 = np.stack([s[:, -2] for s in srt], 1)
arg = np.stack([np.where(X[:, :t].max(1) > 0, X[:, :t].argmax(1), -1) if t else np.full(B, -1) for t in range(10)], 1)
k = np.arange(10)[None, :] - arg
sel = (slice(None), slice(2, 10))
def cols(th):
    return {"x": X[sel], "m1": m1[sel], "m2": m2[sel], "cos": (m1 * np.cos(k * th))[sel], "sin": (m1 * np.sin(k * th))[sel]}
def r2(y, A):
    co = np.linalg.lstsq(A, y, rcond=None)[0]; return 1 - ((y - A @ co) ** 2).mean() / y.var(), co
for n in (1, 6, 7, 8):
    y = H[:, 2:10, n]; on = (y > 0.5) & (m1[sel] > 0) & (arg[sel] >= 0)
    best = max(((r2(y[on], np.c_[tuple(v[on] for v in cols(th).values())])[0], th) for th in np.radians(np.arange(30, 61, 1))))
    th = best[1]; C = cols(th); A = np.c_[tuple(v[on] for v in C.values())]
    full, co = r2(y[on], A)
    drops = {name: full - r2(y[on], np.c_[tuple(v[on] for kk, v in C.items() if kk != name)])[0] for name in C}
    print(f"neuron {n}: on {on.mean():.2f} of steps; best angle {np.degrees(th):.0f} deg; R2 {full:.3f}")
    print("   coefficients  " + "  ".join(f"{kk} {c:6.2f}" for kk, c in zip(C, co)))
    print("   R2 lost when dropped  " + "  ".join(f"{kk} {d:.3f}" for kk, d in drops.items()))
