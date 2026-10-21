"""Diagnostic (not a simulation): which neurons' state do the register-wave formulas for the cancelling neurons miss?
Add the true previous-step values of chosen neurons as extra quantities, fit by least squares, and measure the angle
to the true 4-vector on steps where all four are on, and the accuracy when substituted."""
import numpy as np
import io, contextlib
with contextlib.redirect_stdout(io.StringIO()):
    import register_features as rf
import simlib
from simlib import run

_, _, Hf = simlib.run(rf.Xf); ns = rf.ns


def lagged(Hs, extra):
    out = np.zeros(Hs.shape[:2] + (len(extra),)); out[:, 1:] = Hs[:, :-1][:, :, extra]; return out


for lab, extra in (("none", []), ("neuron 0", [0]), ("neuron 15", [15]), ("neurons 0 and 15", [0, 15]),
                   ("all eight memory neurons", [0, 3, 10, 11, 12, 13, 14, 15]), ("store and detector", [2, 4])):
    Ff = np.concatenate([rf.FX, lagged(Hf, extra)], -1); Fs = np.concatenate([rf.FS, lagged(rf.H, extra)], -1)
    sims = {}
    for n in ns:
        co = np.linalg.lstsq(Ff.reshape(-1, Ff.shape[-1]), rf.PREf[:, :, n].reshape(-1), rcond=None)[0]; sims[n] = np.maximum(Fs @ co, 0)
    F = np.stack([sims[n] for n in ns], -1); V = rf.H[:, :, ns]; m = (V > 0).all(-1)
    c = (F * V).sum(-1)[m] / np.maximum(np.linalg.norm(F, axis=-1)[m] * np.linalg.norm(V, axis=-1)[m], 1e-9)
    acc = (run(rf.X, sims)[0].argmax(1) == rf.y).mean()
    print(f"register waves + previous true values of {lab:26s}: angle (all four on) median {np.degrees(np.arccos(np.clip(np.median(c), -1, 1))):.2f} deg; accuracy {acc:.3f}", flush=True)
