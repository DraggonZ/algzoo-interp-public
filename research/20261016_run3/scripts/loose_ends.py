"""Review 7's goal-1 loose ends.
(a) ARC's model: zero one neuron at every step (3, 5, 9, and 0 and 4 for scale) and measure the accuracy lost (2^22 inputs).
(b) In each new model, the neuron best matching ARC's neuron 4 by probe profile: on full random inputs, does it fire in
    proportion to how much a new input exceeds the running largest, and stay silent otherwise? (ARC's neuron 4: corr 0.78,
    silent 98.8% otherwise.)
(c) For each of the 22 models, the angle between the weights' strongest rotation plane (hide_weights.py) and the activity
    plane of the lone-input response (rotation_check.py, ages 1-9, background -0.32): principal angles in degrees."""
import glob
import sys

import numpy as np
import torch as th

sys.path.insert(0, "/data/output/research/algzoo/20261016_run3/scripts")
from new_mechanism import N4, profiles  # noqa: E402

from algzoo.core import example, load, zoo  # noqa: E402

dev = "cuda"
# (a)
p = example()
w_hi, w_hh, w_oh = (th.as_tensor(np.asarray(q), dtype=th.float32, device=dev) for q in p)
g = th.Generator(device=dev).manual_seed(5)
X = th.randn(2 ** 22, 10, device=dev, generator=g); tgt = X.argsort(1)[:, -2]


def acc(mask):
    h = th.zeros(X.shape[0], 16, device=dev)
    for t in range(10):
        h = th.relu(h @ w_hh.T + X[:, t, None] * w_hi) * mask
    return ((h @ w_oh.T).argmax(1) == tgt).float().mean().item()


base = acc(th.ones(16, device=dev))
line = [f"(a) ARC's model accuracy {100 * base:.2f}%"]
for k in [3, 5, 9, 0, 4]:
    m = th.ones(16, device=dev); m[k] = 0
    line.append(f"without neuron {k}: {100 * (acc(m) - base):+.2f} points")
print("; ".join(line), flush=True)

models = [("example", example())] + [(f"seed{s}", zoo(16, 10, s)) for s in range(5)]
models += [(f.rsplit("_", 1)[1].split(".")[0], load(f)) for f in sorted(glob.glob("/data/output/research/algzoo/20261016_run3/models/2nd_argmax_16_10_1073741824_*.pth"))]
rng = np.random.default_rng(4)
Xn = rng.standard_normal((100000, 10))
print("(b) detector neuron on full inputs (best probe match to ARC's neuron 4):")
for name, p in models:
    wi, wh, wo = (np.asarray(q, np.float64) for q in p)
    k = int(np.argmax(profiles(p) @ N4)); match = float(np.max(profiles(p) @ N4))
    h = np.zeros((len(Xn), 16)); gaps, vals = [], []
    for t in range(10):
        h = np.maximum(0, h @ wh.T + Xn[:, t, None] * wi)
        if t >= 1:
            gaps.append(Xn[:, t] - Xn[:, :t].max(1)); vals.append(h[:, k])
    gg = np.concatenate(gaps); vv = np.concatenate(vals); pos = gg > 0
    print(f"  {name}: neuron {k} (profile match {match:.2f}); corr with excess over running largest {np.corrcoef(vv[pos], gg[pos])[0, 1]:.2f}; "
          f"silent when not a new largest {np.mean(vv[~pos] < 1e-6):.3f}", flush=True)
print("(c) weights' strongest rotation plane against the activity plane (principal angles):")
for name, p in models:
    wi, wh, wo = (np.asarray(q, np.float64) for q in p)
    x = np.full(10, -0.32); x0 = x.copy(); x[0] = 1.5

    def st(xx):
        h = np.zeros(16); H = []; on = []
        for t in range(10):
            h = np.maximum(0, wh @ h + xx[t] * wi); H.append(h); on.append(h > 0)
        return np.array(H), on
    H1, on = st(x); H0, _ = st(x0)
    R = (H1 - H0)[1:10]
    _, _, Vt = np.linalg.svd(R - R.mean(0), full_matrices=False); A = Vt[:2].T
    pat = (np.mean(on[1:10], 0) >= 0.5).astype(float)
    ev, V = np.linalg.eig(pat[:, None] * wh)
    k = [i for i in np.argsort(-np.abs(ev)) if abs(ev[i]) > 0.5 and ev[i].imag > 1e-6][0]
    B, _ = np.linalg.qr(np.c_[V[:, k].real, V[:, k].imag])
    s = np.linalg.svd(A.T @ B, compute_uv=False)
    ang = np.degrees(np.arccos(np.clip(s, -1, 1)))
    print(f"  {name}: weight rotation {np.degrees(abs(np.angle(ev[k]))):.0f} deg/step; principal angles {ang[0]:.0f} and {ang[1]:.0f} degrees", flush=True)
