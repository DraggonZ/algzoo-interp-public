"""Analysis: job 8 (hiding rebuilt each step; the scorer's own measure, re-implemented in numpy) and own accuracy for the
r1a-3 network at a few hiding margins kappa. Also checks that build_v26(delta=0, kappa=0.5) is r1a-3's network.
Question: is the readout's view of the hiding neurons dominated by the wave because the wave is large (K = 1 + kappa)?"""
import numpy as np
from algzoo.handbuilt_v25 import build_v25
from algzoo.handbuilt_v26 import build_v26

ref = build_v25(eps=0.1, theta=0.75 * 0.58).params; new = build_v26().params
print("v26(delta 0, kappa 0.5) == v25:", all(np.allclose(x, y) for x, y in zip(ref, new)))
rng = np.random.default_rng(7); X = rng.standard_normal((400_000, 10))
Xp = X[X.argmax(1) <= 3][:50000]
for kappa in (0.5, 0.35, 0.25, 0.15):
    net = build_v26(kappa=kappa); wi, wh, wo = [np.asarray(p, float) for p in net.params]
    HID = [net.idx[k] for k in ("flip", "clock0", "clock1", "clock2")]
    h = np.zeros((len(X), 16))
    for t in range(10): h = np.maximum(h @ wh.T + X[:, t:t + 1] * wi, 0)
    acc = ((h @ wo.T).argmax(1) == np.argsort(-X, 1)[:, 1]).mean()
    woc = wo[:, HID] - wo[:, HID].mean(0)
    def st(nudge):
        h = np.zeros((len(Xp), 16)); o = []
        for t in range(10):
            h = np.maximum(h @ wh.T + Xp[:, t:t + 1] * wi, 0)
            if nudge and t == 5: h = h.copy(); h[:, HID] *= 0.9
            o.append(h)
        return o
    a, b = st(False), st(True); d6 = b[6][:, HID] - a[6][:, HID]
    left = np.median(np.linalg.norm(d6, axis=1) / np.maximum(np.linalg.norm(a[6][:, HID], axis=1), 1e-9)) / 0.1
    leftr = np.median(np.linalg.norm(d6 @ woc.T, axis=1) / np.maximum(np.linalg.norm(a[6][:, HID] @ woc.T, axis=1), 1e-12)) / 0.1
    print(f"kappa {kappa:.2f}: accuracy {acc:.4f}  job 8: activity {left:.3f}, readout view {leftr:.3f}")
