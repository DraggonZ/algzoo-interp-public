"""v19 checks: kappa = 0 reproduces v18 exactly; then kappa = 0.5, 1, 2 at v18's own gains and at trained scale:
accuracy (200,000 lists), runner-up is the largest (share, median lead), median lead, clipping, accuracy under noise."""
import numpy as np
from algzoo.handbuilt_v18 import build_v18
from algzoo.handbuilt_v19 import build_v19
import simlib

X = simlib.lists(74, 200_000); y = X.argsort(1)[:, -2]; o = X.argsort(1)[:, ::-1]; rng = np.random.default_rng(0)
a18 = [np.asarray(p, float) for p in build_v18().params]; a19 = [np.asarray(p, float) for p in build_v19(kappa=0.0).params]
print("kappa 0 equals v18:", all(np.array_equal(u, v) for u, v in zip(a18, a19)))
age = simlib.task_vars(X)["age"]
for scale, kw in (("v18 gains", {}), ("trained scale", dict(g=10, a=10, w=10, lam=10.0, c=-9.0, eps=0.1))):
    for kappa in (0.0, 0.5, 1.0, 2.0):
        P = tuple(np.asarray(p, float) for p in build_v19(kappa=kappa, **kw).params)
        L, Pre, H = simlib.run(X, params=P); k = L.argmax(1); j = np.argsort(L, 1)[:, -2]; lead = L[np.arange(len(X)), k] - L[np.arange(len(X)), j]
        rl = j == o[:, 0]; off = (Pre[:, :, [3, 4, 5, 6]] <= 0).sum(-1); clip = ((off > 0) & (off < 4))[age >= 1].mean()
        typ = np.array([np.median(H[:, :, n][H[:, :, n] > 0]) if (H[:, :, n] > 0).any() else 1.0 for n in range(16)])
        tol = []
        for s in (1e-3, 1e-2):
            h = np.zeros((50_000, 16))
            for t in range(10):
                h = np.maximum(h @ P[1].T + X[:50_000, t:t + 1] * P[0], 0); h = np.maximum(h + s * typ * rng.standard_normal(h.shape) * (h > 0), 0)
            tol.append(((h @ P[2].T).argmax(1) == y[:50_000]).mean())
        print(f"{scale:13s} kappa {kappa:3.1f}: accuracy {(k == y).mean():.4f}; runner-up is the largest {rl.mean():.3f} (median lead there {np.median(lead[rl]):.3f}); median lead {np.median(lead):.2f}; clipped {clip:.3f}; noise 1e-3 {tol[0]:.3f}, 1e-2 {tol[1]:.3f}", flush=True)
