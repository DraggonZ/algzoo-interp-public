"""v20 robustness: accuracy under noise of 1e-3, 1e-2, 3e-2 x each neuron's typical active value on every neuron at every
step (as gao/noise_tolerance.py), and noise on the timer at the last step only; per-neuron first-order sensitivity
(sensitivity.py). Against the trained network and v19 (trained scale, kappa 0.5, delta 0.02)."""
import sys
import numpy as np
from algzoo.handbuilt_v21 import build_v21 as build_v20, NAMES
from algzoo.handbuilt_v19 import build_v19
import simlib
from sensitivity import sens
kw = dict(arg.split("=") for arg in sys.argv[1:]); kw = {k: float(v) for k, v in kw.items()}
X = simlib.lists(89, 50_000); y = X.argsort(1)[:, -2]; rng = np.random.default_rng(0)
def noisy(P, typ, s, only=None, steps=range(10)):
    h = np.zeros((len(X), len(P[0])))
    for t in range(10):
        h = np.maximum(h @ P[1].T + X[:, t:t + 1] * P[0], 0)
        if t in steps and s:
            e = s * typ * rng.standard_normal(h.shape)
            if only is not None:
                m = np.zeros(len(typ), bool); m[only] = True; e = e * m
            h = np.maximum(h + e * (h > 0), 0)
    return np.mean((h @ P[2].T).argmax(1) == y)
nets = {"trained": ((simlib.WI, simlib.WH, simlib.WO), [1, 6, 7, 8], [str(i) for i in range(16)]),
        "v19 kappa 0.5 delta 0.02": (tuple(np.asarray(p, float) for p in build_v19(g=10, a=10, w=10, lam=10.0, c=-9.0, eps=0.1, kappa=0.5, delta=0.02).params), [3, 4, 5, 6], None),
        "v21": (build_v20(**kw), [5, 6, 7, 8, 9, 10, 11], NAMES)}
for name, (P, T, names) in nets.items():
    P = tuple(np.asarray(p, float) for p in P); P = (P[0].reshape(-1), P[1], P[2])
    H = simlib.run(X, params=P)[2]; typ = np.array([np.median(H[:, :, n][H[:, :, n] > 0]) if (H[:, :, n] > 0).any() else 1.0 for n in range(len(P[0]))])
    out = [f"accuracy {noisy(P, typ, 0):.3f}"] + [f"noise {s:g}: {noisy(P, typ, s):.3f}" for s in (1e-3, 1e-2, 3e-2)] + [f"timer only, last step, 1e-2: {noisy(P, typ, 1e-2, T, [9]):.3f}"]
    print(f"{name}: " + "; ".join(out), flush=True)
    if names:
        sv, _ = sens(P, X[:20_000]); o = np.argsort(-sv)[:8]
        print("   most sensitive: " + "; ".join(f"{names[n]} {sv[n]:.2f}" for n in o))
