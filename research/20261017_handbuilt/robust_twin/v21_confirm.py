"""v21 confirmation: accuracy on 1,000,000 fresh lists (in chunks); 1% noise with three noise seeds on 100,000 lists;
the same for the trained network."""
import numpy as np
from algzoo.handbuilt_v21 import build_v21
import simlib
P21 = tuple(np.asarray(p, float) for p in build_v21()); PT = (simlib.WI, simlib.WH, simlib.WO)
for name, P in (("v21", P21), ("trained", PT)):
    acc = []
    for k in range(5):
        X = simlib.lists(1000 + k, 200_000); acc.append(np.mean(simlib.run(X, params=P)[0].argmax(1) == X.argsort(1)[:, -2]))
    X = simlib.lists(2000, 100_000); y = X.argsort(1)[:, -2]; H = simlib.run(X, params=P)[2]
    typ = np.array([np.median(H[:, :, n][H[:, :, n] > 0]) if (H[:, :, n] > 0).any() else 1.0 for n in range(len(P[0]))])
    nz = []
    for seed in range(3):
        rng = np.random.default_rng(seed); h = np.zeros((len(X), len(P[0])))
        for t in range(10):
            h = np.maximum(h @ P[1].T + X[:, t:t + 1] * P[0], 0); h = np.maximum(h + 0.01 * typ * rng.standard_normal(h.shape) * (h > 0), 0)
        nz.append(np.mean((h @ P[2].T).argmax(1) == y))
    print(f"{name}: accuracy on 1,000,000 lists {np.mean(acc):.4f}; under 1% noise (three seeds, 100,000 lists) " + ", ".join(f"{v:.3f}" for v in nz), flush=True)
