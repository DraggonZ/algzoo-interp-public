"""Question: after a partial switch, does the hiding group's record of the largest recover? Lists with a largest at
r = 2 that stays largest and a positive number at step 3 that partly switches the hiding group. Counterfactual: the
same list with that number replaced by 0 (no partial switch). For steps 3-9: angle between the hiding group's
sensitivity to the largest (d hiding / d x_r, by nudging) in the real and the counterfactual list, and the ratio of
their sizes; median over lists. Trained network; v18 at g = 10."""
import numpy as np
from algzoo.core import example
from algzoo.handbuilt_v18 import build_v18
rng = np.random.default_rng(12); X = rng.standard_normal((1500000, 10))
def states(P, X):
    P = [np.asarray(p, float) for p in P]; h = np.zeros((len(X), len(P[0]))); hs = []
    for t in range(10): h = np.maximum(h @ P[1].T + X[:, t:t + 1] * P[0], 0); hs.append(h)
    return np.stack(hs, 1)
I = build_v18().idx; hid18 = [I[k] for k in ("clock0", "clock1", "clock2", "flip")]
nets = {"trained": (example(), [1, 6, 7, 8]), "v18 g 10": (build_v18(g=10, a=10, w=10, lam=10.0, c=-9.0, eps=0.1).params, hid18)}
r, c = 2, 3; eps = 1e-3
base = (X[:, r] > np.maximum(X[:, :r].max(1), 0) + 0.05) & (X[:, r] > 0.3) & (X[:, r + 1:].max(1) < X[:, r]) & (X[:, c] > 0)
for name, (P, hid) in nets.items():
    H = states(P, X[base]); off = (H[:, c, hid] == 0).sum(1); part = (off > 0) & (off < len(hid))
    Xs = X[base][part][:40000]; Xc = Xs.copy(); Xc[:, c] = 0.0
    def sens(Xa):
        Xp = Xa.copy(); Xp[:, r] += eps; return (states(P, Xp) - states(P, Xa))[:, :, hid] / eps
    S, Sc = sens(Xs), sens(Xc)
    out = []
    for t in range(c, 10):
        a, b = S[:, t], Sc[:, t]
        cosv = (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-12)
        out.append(f"{t}: {np.degrees(np.arccos(np.clip(np.median(cosv), -1, 1))):5.1f} deg, x{np.median(np.linalg.norm(a, axis=1) / (np.linalg.norm(b, axis=1) + 1e-12)):.2f}")
    print(f"{name:9s} (n={len(Xs)}, partial at step 3 on {part.mean():.2f} of such lists)  " + " | ".join(out))
