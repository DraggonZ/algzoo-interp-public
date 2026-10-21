"""Which neuron makes the twin fragile? Twin with the trained network's gains and memory level 1 (robust_twin.py),
noise of 1e-3 x typical value on one neuron at a time (active steps only). Also the trained network, for comparison."""
import numpy as np
import simlib
from algzoo.handbuilt_v18 import build_v18

X = simlib.lists(1111, 50_000); y = X.argsort(1)[:, -2]; rng = np.random.default_rng(0)
for name, P in (("twin, trained gains, lam 1", tuple(np.asarray(p, float) for p in build_v18(g=10, a=10, w=10, lam=1.0).params)),
                ("trained", (simlib.WI, simlib.WH, simlib.WO))):
    L, _, H = simlib.run(X, params=P)
    typ = np.array([np.median(H[:, :, n][H[:, :, n] > 0]) if (H[:, :, n] > 0).any() else 1.0 for n in range(16)])
    out = []
    for n in range(16):
        h = np.zeros((len(X), 16)); f = np.zeros(16); f[n] = 1
        for t in range(10):
            h = np.maximum(h @ P[1].T + X[:, t:t + 1] * P[0], 0); h = np.maximum(h + f * 1e-3 * typ * rng.standard_normal(h.shape) * (h > 0), 0)
        out.append(((h @ P[2].T).argmax(1) == y).mean())
    print(f"{name}: accuracy {(L.argmax(1) == y).mean():.3f}; noise 1e-3 on one neuron: " + " ".join(f"{n}:{a:.3f}" for n, a in enumerate(out)))
    print("   typical values: " + " ".join(f"{v:.3g}" for v in typ))
