"""Size against kind on the robust twin v21. The timer (45-degree clock C0-2, 90-degree clock E0-2, flip F) written as
closed-form rules of the input, substituted for the network's own timer neurons (everything else runs as built), and
spoiled two ways:
  misreadings: the wave's height taken from the largest m1 instead of the record's jump d; the flip left out;
  slightly off: the 45-degree clock turning 45.5 degrees per step; the wave's height 2% too large (K -> 1.02 K);
plus the store's values 2% too large (store substituted, timer exact). Accuracy on 100,000 lists, and accuracy under
1% noise for scale."""
import numpy as np
from algzoo.handbuilt import FROM_PQ
from algzoo.handbuilt_v21 import build_v21, NAMES
import simlib

X = simlib.lists(92, 100_000); y = X.argsort(1)[:, -2]; B = len(X)
P = tuple(np.asarray(p, float) for p in build_v21()); I = {k: i for i, k in enumerate(NAMES)}
K, s = 1.5, 10.0
m1 = np.zeros((B, 11)); rec = np.zeros((B, 10), bool); d = np.zeros((B, 10))
for t in range(10):
    m1p = m1[:, t]; x = X[:, t]; rec[:, t] = x > m1p; d[:, t] = np.maximum(x - m1p, 0); m1[:, t + 1] = np.maximum(m1p, x)
# m1[:, t] = m1(t-1) (before input t)

def rot(th, n):
    return np.stack([np.cos(th * n), np.sin(th * n)], -1)

def timer(height="jump", deg45=45.0, gain=1.0, flip=True):
    C = np.zeros((B, 10, 3)); E = np.zeros((B, 10, 3)); F = np.zeros((B, 10))
    rho = np.full(B, -1); hd = np.zeros(B)
    for t in range(10):
        if t >= 2:
            new = rec[:, t - 2]; rho = np.where(new, t - 2, rho)
            hd = np.where(new, d[:, t - 2] if height == "jump" else m1[:, t - 1], hd)   # m1[:, t-1] = value of the record at t-2
        live = (rho >= 0) & ~(rec[:, t - 1] if t >= 1 else np.zeros(B, bool))
        age = t - rho - 2
        L = K * m1[:, t]
        z1 = gain * K * hd[:, None] * rot(np.radians(deg45), age)
        z2 = gain * K * hd[:, None] * rot(np.pi / 2, age)
        zf = gain * K * hd * (-1.0) ** age * (1.0 if flip else 0.0)
        C[:, t] = np.where(live[:, None], np.maximum(L[:, None] + z1 @ FROM_PQ.T, 0), 0)
        E[:, t] = np.where(live[:, None], np.maximum(L[:, None] + z2 @ FROM_PQ.T, 0), 0)
        F[:, t] = np.where(live, np.maximum(L + zf, 0), 0)
        # before any injection (rho < 0) and not wiped: level only
        none = (rho < 0) & ~(rec[:, t - 1] if t >= 1 else np.zeros(B, bool))
        C[:, t] = np.where(none[:, None], L[:, None], C[:, t]); E[:, t] = np.where(none[:, None], L[:, None], E[:, t]); F[:, t] = np.where(none, L, F[:, t])
    return {**{I[f"C{k}"]: C[:, :, k] for k in range(3)}, **{I[f"E{k}"]: E[:, :, k] for k in range(3)}, I["F"]: F}

H = simlib.run(X, params=P)[2]
ex = timer()
err = max(np.max(np.abs(ex[n] - H[:, :, n])) for n in ex)
print(f"exact timer rules against the network's own timer values: largest difference {err:.2e}")
acc = lambda sims: np.mean(simlib.run(X, sims, params=P)[0].argmax(1) == y)
print(f"network {acc({}):.3f}; exact timer rules {acc(ex):.3f}")
for lab, kw in (("misreading: height from the largest", dict(height="largest")), ("misreading: flip left out", dict(flip=False)),
                ("slightly off: 45-degree clock at 45.5 degrees", dict(deg45=45.5)), ("slightly off: wave height 2% too large", dict(gain=1.02))):
    print(f"{lab:46s}: {acc(timer(**kw)):.3f}", flush=True)
print(f"{'slightly off: store 2% too large':46s}: {acc({I['S']: 1.02 * H[:, :, I['S']]}):.3f}")
