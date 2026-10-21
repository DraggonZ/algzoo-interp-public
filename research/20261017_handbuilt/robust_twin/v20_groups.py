"""v20 (lev 1.0, lam 12.15): accuracy with 1% noise on one group of neurons at every step."""
import numpy as np
from algzoo.handbuilt_v20 import build_v20, NAMES
import simlib
X = simlib.lists(91, 50_000); y = X.argsort(1)[:, -2]; rng = np.random.default_rng(0)
P = build_v20(lev=1.0, lam=12.15); H = simlib.run(X, params=P)[2]
typ = np.array([np.median(H[:, :, n][H[:, :, n] > 0]) if (H[:, :, n] > 0).any() else 1.0 for n in range(len(NAMES))])
I = {k: i for i, k in enumerate(NAMES)}
groups = {"store": ["S"], "detector": ["D"], "carrier+flag": ["Zc", "Z"], "keep gate": ["G"], "timer": ["C0", "C1", "C2", "F"],
          "reveal": ["QC0", "QC1", "QC2", "QF"], "memory": [f"Y{j}" for j in range(9)], "all": NAMES}
for g, ns in groups.items():
    m = np.zeros(len(NAMES), bool); m[[I[k] for k in ns]] = True; h = np.zeros((len(X), len(NAMES)))
    for t in range(10):
        h = np.maximum(h @ P[1].T + X[:, t:t + 1] * P[0], 0); h = np.maximum(h + 0.01 * typ * rng.standard_normal(h.shape) * (h > 0) * m, 0)
    print(f"1% noise on {g:13s}: accuracy {np.mean((h @ P[2].T).argmax(1) == y):.3f}", flush=True)
