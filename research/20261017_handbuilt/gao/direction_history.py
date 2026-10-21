"""Is the cancellers' missing history their own partial switching? Register-wave formulas (fit on active steps).
On steps with all four cancelling neurons on, split by whether, at any earlier step of the list, some but not all of
the four were on ('partly switched before'). Median angle to the true 4-vector in each part."""
import numpy as np
import io, contextlib
with contextlib.redirect_stdout(io.StringIO()):
    import register_features as rf
import simlib

_, _, Hf = simlib.run(rf.Xf); ns = rf.ns; sims = {}
for n in ns:
    m = Hf[:, :, n].reshape(-1) > 0
    co = np.linalg.lstsq(rf.FX.reshape(-1, rf.FX.shape[-1])[m], rf.PREf[:, :, n].reshape(-1)[m], rcond=None)[0]; sims[n] = np.maximum(rf.FS @ co, 0)
F = np.stack([sims[n] for n in ns], -1); V = rf.H[:, :, ns]; on = (V > 0).sum(-1)
partial = (on > 0) & (on < 4); before = np.zeros_like(partial)
for t in range(1, 10):
    before[:, t] = before[:, t - 1] | partial[:, t - 1]
allon = on == 4
c = (F * V).sum(-1) / np.maximum(np.linalg.norm(F, axis=-1) * np.linalg.norm(V, axis=-1), 1e-9)
for lab, m in (("never partly switched before", allon & ~before), ("partly switched at some earlier step", allon & before)):
    print(f"all four on, {lab:38s}: {m.sum() / allon.sum():.2f} of such steps; angle median {np.degrees(np.arccos(np.clip(np.median(c[m]), -1, 1))):.2f} deg, 90th percentile {np.degrees(np.arccos(np.clip(np.percentile(c[m], 10), -1, 1))):.2f}")
