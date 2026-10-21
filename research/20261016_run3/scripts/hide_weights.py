"""Is the output blind to the recurrent matrix's rotating plane? (goal 1, from the weights given an on/off pattern)

Pattern = which neurons are on most often along the lone-input response (ages 1-9, background -0.32). Step matrix
M = diag(pattern) W_hh. Take its complex eigenvector pairs with |eigenvalue| > 0.5 (the rotations), and measure how much of
each rotation plane the centred readout can see: ||Wc V|| / ||Wc|| for an orthonormal basis V of the plane, compared with
random 2-D planes inside the on-neurons. A visibility far below random means the readout is built blind to the rotation."""
import glob

import numpy as np

from algzoo.core import example, load, zoo

models = [("example", example())] + [(f"seed{s}", zoo(16, 10, s)) for s in range(5)]
models += [(f.rsplit("_", 1)[1].split(".")[0], load(f)) for f in sorted(glob.glob("/data/output/research/algzoo/20261016_run3/models/2nd_argmax_16_10_1073741824_*.pth"))]
rng = np.random.default_rng(0)
rows = []
for name, p in models:
    w_hi, w_hh, w_oh = (np.asarray(q, np.float64) for q in p)
    d = len(w_hi)
    x = np.full(10, -0.32); x[0] = 1.5
    h = np.zeros(d); on = []
    for t in range(10):
        h = np.maximum(0, w_hh @ h + x[t] * w_hi); on.append(h > 0)
    pat = (np.mean(on[1:10], 0) >= 0.5).astype(float)
    M = pat[:, None] * w_hh
    ev, V = np.linalg.eig(M)
    Wc = w_oh - w_oh.mean(0)
    vis = []
    for k in np.argsort(-np.abs(ev)):
        if abs(ev[k]) > 0.5 and ev[k].imag > 1e-6:
            B, _ = np.linalg.qr(np.c_[V[:, k].real, V[:, k].imag])
            vis.append((abs(ev[k]), np.degrees(abs(np.angle(ev[k]))), np.linalg.norm(Wc @ B) / np.sqrt(2)))
    idx = np.flatnonzero(pat)
    rand = []
    for _ in range(2000):
        B = np.zeros((d, 2)); B[idx] = rng.standard_normal((len(idx), 2)); B, _ = np.linalg.qr(B)
        rand.append(np.linalg.norm(Wc @ B) / np.sqrt(2))
    rand = np.array(rand)
    if vis:
        top = vis[0]
        pct = np.mean(rand <= top[2])
        rows.append((top[2] / np.median(rand), pct))
        print(f"{name}: strongest rotation |eig| {top[0]:.2f} at {top[1]:.0f} deg/step; readout sees {top[2]:.2f} of it per unit, random planes median {np.median(rand):.2f} "
              f"(ratio {top[2] / np.median(rand):.2f}; random planes as blind or blinder: {pct:.1%}); other rotations: " + ", ".join(f"{a:.0f} deg {c / np.median(rand):.2f}" for _, a, c in vis[1:3]), flush=True)
    else:
        print(f"{name}: no complex eigenvalue above 0.5 for its main on/off pattern", flush=True)
R = np.array(rows)
print(f"models with a rotation: {len(R)}; readout visibility of the strongest rotation relative to random planes: median {np.median(R[:, 0]):.2f}, "
      f"below 0.5 in {np.sum(R[:, 0] < 0.5)}; random planes as blind or blinder in at most {R[:, 1].max():.1%}")
