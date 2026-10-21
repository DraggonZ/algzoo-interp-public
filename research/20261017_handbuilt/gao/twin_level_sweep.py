"""Within the hand-built family: shrink the timer level g (clock and flip) of the trained-scale twin (a = w = 10,
lam 10, vote gain 10, tie-breaker 1% of a vote) and measure accuracy, switching rate (on/off changes on steps without a
new largest), lead / pushes, and accuracy under noise of 1e-3 and 1e-2 x typical values. Prediction from
switch_vs_cancel.log: smaller g, more switching, less cancellation, better noise tolerance relative to accuracy."""
import numpy as np
from algzoo.handbuilt_v18 import build_v18
import simlib

X = simlib.lists(61, 50_000); y = X.argsort(1)[:, -2]; rng = np.random.default_rng(0); V = simlib.task_vars(X)
quiet = (V["newmax"] == 0); quiet[:, :2] = False
for g in (30, 10, 3, 1, 0.3):
    P = tuple(np.asarray(p, float) for p in build_v18(g=g, a=10, w=10, lam=10.0, c=-9.0, eps=0.1).params)
    L, _, H = simlib.run(X, params=P); k = L.argmax(1); j = np.argsort(L, 1)[:, -2]
    lead = L[np.arange(len(L)), k] - L[np.arange(len(L)), j]; push = np.abs((P[2][k] - P[2][j]) * H[:, 9]).sum(1); m = push > 1e-9
    on = H > 0; sw = (on[:, 1:] != on[:, :-1])[quiet[:, 1:]].mean()
    typ = np.array([np.median(H[:, :, n][H[:, :, n] > 0]) if (H[:, :, n] > 0).any() else 1.0 for n in range(16)])
    tol = []
    for s in (1e-3, 1e-2):
        h = np.zeros((len(X), 16))
        for t in range(10):
            h = np.maximum(h @ P[1].T + X[:, t:t + 1] * P[0], 0); h = np.maximum(h + s * typ * rng.standard_normal(h.shape) * (h > 0), 0)
        tol.append(((h @ P[2].T).argmax(1) == y).mean())
    acc = (k == y).mean()
    print(f"timer level g {g:4g}: accuracy {acc:.3f}; switching rate {sw:.3f}; lead/pushes {np.median(lead[m] / push[m]):.4f}; noise 1e-3 {tol[0]:.3f}, 1e-2 {tol[1]:.3f} (kept {tol[1] / acc:.2f} of accuracy)", flush=True)
