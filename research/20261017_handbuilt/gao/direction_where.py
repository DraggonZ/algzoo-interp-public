"""Where do the formulas for the cancelling neurons miss the direction? Register-wave formulas (register_features.py).
Angle between formula and true 4-vector, split by how many of the four neurons are on, and the accuracy when the true
values are used on steps where all four are on and the formulas elsewhere (and the reverse)."""
import numpy as np
import io, contextlib
with contextlib.redirect_stdout(io.StringIO()):
    import register_features as rf
from simlib import run

ns = rf.ns; sims = {}
for n in ns:
    co = np.linalg.lstsq(rf.FX.reshape(-1, rf.FX.shape[-1]), rf.PREf[:, :, n].reshape(-1), rcond=None)[0]; sims[n] = np.maximum(rf.FS @ co, 0)
F = np.stack([sims[n] for n in ns], -1); V = rf.H[:, :, ns]; on = (V > 0).sum(-1)
nf = np.linalg.norm(F, axis=-1); nv = np.linalg.norm(V, axis=-1)
cosang = (F * V).sum(-1) / np.maximum(nf * nv, 1e-9)
for k in range(1, 5):
    m = (on == k) & (nv > 1e-6)
    print(f"{k} of 4 on: {m.mean():.2f} of steps; angle median {np.degrees(np.arccos(np.clip(np.median(cosang[m]), -1, 1))):.1f} deg")
m0 = on == 0; print(f"all off: {m0.mean():.2f} of steps; formulas give a nonzero pattern on {np.mean(nf[m0] > 1e-6):.2f} of them (median size {np.median(nf[m0]):.2f})")
allon = (on == 4)[..., None]
mixA = np.where(allon, V, F); mixB = np.where(allon, F, V)
for lab, M in (("true where all four on, formulas elsewhere", mixA), ("formulas where all four on, true elsewhere", mixB)):
    print(f"{lab}: accuracy {(run(rf.X, {n: M[:, :, i] for i, n in enumerate(ns)})[0].argmax(1) == rf.y).mean():.3f}", flush=True)
