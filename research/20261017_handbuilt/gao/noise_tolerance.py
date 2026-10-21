"""How exact must an explanation be? Replace all 16 neurons, at every step, by their own true values plus Gaussian noise
of s x each neuron's typical active value (median positive activation), then ReLU; the noise feeds forward through the
recurrence as a simulation's error would. Trained network and the twin (gains 1e4 and 30). 100,000 lists."""
import numpy as np
import simlib
from simlib import lists, score
from algzoo.handbuilt_v18 import build_v18

X = lists(707, 100_000); y = X.argsort(1)[:, -2]
nets = {"trained": (simlib.WI, simlib.WH, simlib.WO)}
for g, w, a in ((1e4, 1e4, 1000.0), (30.0, 30.0, 20.0)):
    P = tuple(np.asarray(p, float) for p in build_v18(g=g, w=w, a=a).params); nets[f"twin gains {g:g}"] = (P[0], P[1], P[2] * 447.7)
rng = np.random.default_rng(0)
for name, P in nets.items():
    L, _, H = simlib.run(X, params=P)
    typ = np.array([np.median(H[:, :, n][H[:, :, n] > 0]) if (H[:, :, n] > 0).any() else 1.0 for n in range(16)])
    row = []
    for s in (1e-8, 1e-6, 1e-4, 1e-3, 1e-2, 3e-2, 1e-1):
        wi, wh, wo = P; h = np.zeros((len(X), 16))
        for t in range(10):
            h = np.maximum(h @ wh.T + X[:, t:t + 1] * wi, 0)
            h = np.maximum(h + s * typ * rng.standard_normal(h.shape) * (h > 0), 0)  # noise on active values only
        row.append(f"{s:g}: {((h @ wo.T).argmax(1) == y).mean():.3f}")
    print(f"{name:16s} acc {(L.argmax(1) == y).mean():.3f}; with noise " + ", ".join(row), flush=True)
