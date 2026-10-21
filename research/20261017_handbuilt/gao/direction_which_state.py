"""Diagnostic: which neurons' previous values, given exactly alongside the register waves, close the cancelling
neurons' direction gap? Least squares on active steps; angle on steps with all four on; accuracy when substituted."""
import numpy as np
import io, contextlib
with contextlib.redirect_stdout(io.StringIO()):
    import register_features as rf
import simlib
from simlib import run

_, _, Hf = simlib.run(rf.Xf); ns = rf.ns


def lagged(Hs, extra):
    out = np.zeros(Hs.shape[:2] + (len(extra),)); out[:, 1:] = Hs[:, :-1][:, :, extra]; return out


for lab, extra in (("none", []), ("the four cancelling neurons", [1, 6, 7, 8]), ("cancelling + store/detector", [1, 6, 7, 8, 2, 4]),
                   ("everything but the cancelling neurons", [0, 2, 3, 4, 5, 9, 10, 11, 12, 13, 14, 15]), ("all 16", list(range(16)))):
    Ff = np.concatenate([rf.FX, lagged(Hf, extra)], -1); Fs = np.concatenate([rf.FS, lagged(rf.H, extra)], -1); sims = {}
    for n in ns:
        m = Hf[:, :, n].reshape(-1) > 0
        co = np.linalg.lstsq(Ff.reshape(-1, Ff.shape[-1])[m], rf.PREf[:, :, n].reshape(-1)[m], rcond=None)[0]; sims[n] = np.maximum(Fs @ co, 0)
    F = np.stack([sims[n] for n in ns], -1); V = rf.H[:, :, ns]; mm = (V > 0).all(-1)
    c = (F * V).sum(-1)[mm] / np.maximum(np.linalg.norm(F, axis=-1)[mm] * np.linalg.norm(V, axis=-1)[mm], 1e-9)
    print(f"register waves + true previous {lab:38s}: angle median {np.degrees(np.arccos(np.clip(np.median(c), -1, 1))):.2f} deg; accuracy {(run(rf.X, sims)[0].argmax(1) == rf.y).mean():.3f}", flush=True)
