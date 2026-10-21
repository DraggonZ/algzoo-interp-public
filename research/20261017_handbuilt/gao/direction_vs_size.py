"""Is our account of the cancelling neurons wrong in direction or in size? Per list and step, the true values v of
neurons 1, 6, 7, 8 and our fitted formulas' values f (account_features, least squares, as matched_size.py). Variants
substituted for the four neurons: (a) f itself; (b) size only wrong: v scaled to f's size (|f| / |v| x v); (c)
direction only wrong: f rescaled to v's size (|v| / |f| x f). Also the angle between f and v. 50,000 lists."""
import numpy as np
from matched_size import X, y, L, H, fitted, account_features
from simlib import run

ns = [1, 6, 7, 8]
sims = fitted("cancellers", account_features)
F = np.stack([sims[n] for n in ns], -1); V = H[:, :, ns]
nf = np.linalg.norm(F, axis=-1, keepdims=True); nv = np.linalg.norm(V, axis=-1, keepdims=True)
size_only = np.where(nv > 1e-9, V * nf / np.maximum(nv, 1e-9), F)
dir_only = np.where(nf > 1e-9, F * nv / np.maximum(nf, 1e-9), V)
cos = (F * V).sum(-1) / np.maximum(nf[..., 0] * nv[..., 0], 1e-9); act = nv[..., 0] > 1e-6
print(f"angle between formula and true pattern (active steps): median {np.degrees(np.arccos(np.clip(np.median(cos[act]), -1, 1))):.1f} degrees, "
      f"90th percentile {np.degrees(np.arccos(np.clip(np.percentile(cos[act], 10), -1, 1))):.1f}; size ratio |f|/|v| median {np.median((nf / np.maximum(nv, 1e-9))[..., 0][act]):.3f}, "
      f"10-90% {np.percentile((nf / np.maximum(nv, 1e-9))[..., 0][act], 10):.3f}-{np.percentile((nf / np.maximum(nv, 1e-9))[..., 0][act], 90):.3f}")
for lab, M in (("formulas as they are", F), ("size only wrong (true direction)", size_only), ("direction only wrong (true size)", dir_only)):
    acc = (run(X, {n: M[:, :, i] for i, n in enumerate(ns)})[0].argmax(1) == y).mean()
    print(f"{lab:34s}: accuracy {acc:.3f}", flush=True)

# How exact must the direction be? Rotate the true pattern v by a fixed angle toward a random direction orthogonal to
# it (new list and step each time), keeping its size, and clip at zero as a ReLU would.
rng = np.random.default_rng(0)
for deg in (0.5, 1, 2, 5, 10):
    Z = rng.standard_normal(V.shape); Z -= (Z * V).sum(-1, keepdims=True) * V / np.maximum(nv ** 2, 1e-12)
    Z /= np.maximum(np.linalg.norm(Z, axis=-1, keepdims=True), 1e-12)
    th = np.radians(deg); R = np.maximum(V * np.cos(th) + Z * nv * np.sin(th), 0)
    acc = (run(X, {n: R[:, :, i] for i, n in enumerate(ns)})[0].argmax(1) == y).mean()
    print(f"true pattern turned {deg:4.1f} degrees: accuracy {acc:.3f}", flush=True)
