"""Fit target matters: least squares on pre-activations over all steps (as so far) against only the steps where the
neuron is on (its negative values do not reach the output, as long as the formula keeps them negative). Cancelling
neurons, our account and the register waves (with and without the true previous memory state, as a diagnostic)."""
import numpy as np
import io, contextlib
with contextlib.redirect_stdout(io.StringIO()):
    import register_features as rf
import simlib
from simlib import run
from sim_groups import account_features

_, _, Hf = simlib.run(rf.Xf); ns = rf.ns


def lagged(Hs, extra):
    out = np.zeros(Hs.shape[:2] + (len(extra),)); out[:, 1:] = Hs[:, :-1][:, :, extra]; return out


def go(lab, Ff_fn, Fs_fn):
    for active in (False, True):
        sims = {}
        for n in ns:
            Ff = Ff_fn(n); A = Ff.reshape(-1, Ff.shape[-1]); t = rf.PREf[:, :, n].reshape(-1)
            m = (Hf[:, :, n].reshape(-1) > 0) if active else np.ones(len(t), bool)
            co = np.linalg.lstsq(A[m], t[m], rcond=None)[0]; sims[n] = np.maximum(Fs_fn(n) @ co, 0)
        F = np.stack([sims[n] for n in ns], -1); V = rf.H[:, :, ns]; mm = (V > 0).all(-1)
        c = (F * V).sum(-1)[mm] / np.maximum(np.linalg.norm(F, axis=-1)[mm] * np.linalg.norm(V, axis=-1)[mm], 1e-9)
        acc = (run(rf.X, sims)[0].argmax(1) == rf.y).mean()
        print(f"{lab:40s} fit on {'active steps' if active else 'all steps   '}: angle median {np.degrees(np.arccos(np.clip(np.median(c), -1, 1))):.2f} deg; accuracy {acc:.3f}", flush=True)


go("our account (5)", lambda n: account_features(rf.Xf, n), lambda n: account_features(rf.X, n))
go("register waves (23)", lambda n: rf.FX, lambda n: rf.FS)
E = [0, 3, 10, 11, 12, 13, 14, 15]
go("register waves + true memory state", lambda n: np.concatenate([rf.FX, lagged(Hf, E)], -1), lambda n: np.concatenate([rf.FS, lagged(rf.H, E)], -1))
