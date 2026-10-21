"""What do ARC's four cancelling neurons (1, 6, 7, 8) hold? On random lists, after each step, fit each neuron's
activity (only where it is on) by least squares on candidate features, adding them in groups. Also: how many
directions their joint activity uses."""
import numpy as np
from algzoo.core import example

wi, wh, wo = (np.asarray(w, dtype=float) for w in example()); wi = wi.reshape(-1)
canc = [1, 6, 7, 8]
rng = np.random.default_rng(16)
X = rng.standard_normal((200_000, 10)); B, n = X.shape
h = np.zeros((B, 16)); rows = []
m1 = np.zeros(B); m2 = np.zeros(B); Lpos = np.full(B, -1)
for t in range(n):
    x = X[:, t]
    h = np.maximum(h @ wh.T + x[:, None] * wi, 0)
    rec = x > m1
    m2 = np.where(rec, m1, np.maximum(m2, np.minimum(x, m1)))
    m1 = np.maximum(m1, x); Lpos = np.where(rec, t, Lpos)
    age = np.where(Lpos >= 0, t - Lpos, 0)
    ang = np.deg2rad(40.0) * age
    F = {"m1": m1, "m2": m2, "x": x, "relu_x": np.maximum(x, 0),
         "m1cos": m1 * np.cos(ang), "m1sin": m1 * np.sin(ang), "m1cos2": m1 * np.cos(2 * ang), "m1sin2": m1 * np.sin(2 * ang)}
    if t >= 2:
        rows.append((h[:, canc].copy(), F))
H = np.concatenate([r[0] for r in rows]); Fall = {k: np.concatenate([r[1][k] for r in rows]) for k in rows[0][1]}
groups = [("largest so far", ["m1"]), ("+ timer angle", ["m1", "m1cos", "m1sin"]),
          ("+ second-largest so far", ["m1", "m1cos", "m1sin", "m2"]),
          ("+ last input", ["m1", "m1cos", "m1sin", "m2", "x", "relu_x"]),
          ("+ second harmonic of the angle", ["m1", "m1cos", "m1sin", "m2", "x", "relu_x", "m1cos2", "m1sin2"]),
          ("angle + last input, no second-largest", ["m1", "m1cos", "m1sin", "x", "relu_x"])]
print("share of variance explained (R^2) where the neuron is on; columns: neurons 1, 6, 7, 8")
for name, keys in groups:
    out = []
    for k, nrn in enumerate(canc):
        on = H[:, k] > 0
        A = np.c_[[Fall[f][on] for f in keys]].T
        y = H[on, k]
        coef, *_ = np.linalg.lstsq(A, y, rcond=None)
        out.append(1 - ((A @ coef - y) ** 2).sum() / ((y - y.mean()) ** 2).sum())
    print(f"  {name:40s} " + "  ".join(f"{r:.3f}" for r in out))
print("share of time on:", np.round((H > 0).mean(0), 2))
S = np.linalg.svd(H - H.mean(0), compute_uv=False)
print("directions of joint activity, share of variance:", np.round(S ** 2 / (S ** 2).sum(), 3))

print("\nafter removing the last input and the largest so far: share of the REMAINING variance explained")
base = ["x", "relu_x", "m1"]
for name, extra in (("timer angle", ["m1cos", "m1sin"]), ("second-largest so far", ["m2"]),
                    ("angle + second-largest", ["m1cos", "m1sin", "m2"]),
                    ("angle + second-largest + 2nd harmonic", ["m1cos", "m1sin", "m2", "m1cos2", "m1sin2"])):
    out = []
    for k in range(4):
        on = H[:, k] > 0; y = H[on, k]
        A0 = np.c_[[Fall[f][on] for f in base]].T
        r0 = y - A0 @ np.linalg.lstsq(A0, y, rcond=None)[0]
        A = np.c_[[Fall[f][on] for f in base + extra]].T
        r = y - A @ np.linalg.lstsq(A, y, rcond=None)[0]
        out.append(1 - (r ** 2).sum() / (r0 ** 2).sum())
    print(f"  {name:40s} " + "  ".join(f"{v:.2f}" for v in out))
on_all = (H > 0).all(1)
A0 = np.c_[[Fall[f][on_all] for f in base]].T
R = H[on_all] - A0 @ np.linalg.lstsq(A0, H[on_all], rcond=None)[0]
S = np.linalg.svd(R - R.mean(0), compute_uv=False)
print("directions of what is left (all four on), share:", np.round(S ** 2 / (S ** 2).sum(), 3))
