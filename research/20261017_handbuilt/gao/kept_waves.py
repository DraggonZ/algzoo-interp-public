"""Do the kept numbers since the current largest supply the cancelling neurons' missing history? For each rotation pair
(angle th, size rho) from the weights (register_features.py), add the wave of kept numbers that arrived after the
current largest: sum over such steps s of kv(s) rho^(t-s) cos((t-s) th) and the sine (kv = the number if kept by
the account's gate rule, x > 0 and x >= 0.75 x second-largest, else 0). Also the same over all kept numbers (no wipe),
as a control for 'since the largest'. Least squares on active steps; angle (all four on) and accuracy."""
import numpy as np
import io, contextlib
with contextlib.redirect_stdout(io.StringIO()):
    import register_features as rf
import simlib
from simlib import run, task_vars

_, _, Hf = simlib.run(rf.Xf); ns = rf.ns


def kept_waves(Xs, since_largest=True, every_input=False):
    V = task_vars(Xs); x, m2 = V["x"], V["m2"]; B, T = Xs.shape
    kv = x if every_input else np.where((x > 0) & (x >= 0.75 * m2) & ~(V["newmax"] > 0), x, 0)
    pos = np.full(B, -1); posl = np.zeros((B, T), int)
    for t in range(T):
        pos = np.where(V["newmax"][:, t] > 0, t, pos); posl[:, t] = pos
    out = []
    for th, rho in rf.pairs:
        c = np.zeros((B, T)); s_ = np.zeros((B, T))
        for t in range(T):
            for s in range(t + 1):
                ok = (s > posl[:, t]) if since_largest else np.ones(B, bool)
                a = t - s; w = kv[:, s] * ok * rho ** a
                c[:, t] += w * np.cos(a * th); s_[:, t] += w * np.sin(a * th)
        out += [c, s_]
    return np.stack(out, -1)


def go(lab, Ff, Fs):
    sims = {}
    for n in ns:
        m = Hf[:, :, n].reshape(-1) > 0
        co = np.linalg.lstsq(Ff.reshape(-1, Ff.shape[-1])[m], rf.PREf[:, :, n].reshape(-1)[m], rcond=None)[0]; sims[n] = np.maximum(Fs @ co, 0)
    F = np.stack([sims[n] for n in ns], -1); V = rf.H[:, :, ns]; mm = (V > 0).all(-1)
    c = (F * V).sum(-1)[mm] / np.maximum(np.linalg.norm(F, axis=-1)[mm] * np.linalg.norm(V, axis=-1)[mm], 1e-9)
    print(f"{lab:52s} ({Ff.shape[-1]:2d} quantities): angle median {np.degrees(np.arccos(np.clip(np.median(c), -1, 1))):.2f} deg; accuracy {(run(rf.X, sims)[0].argmax(1) == rf.y).mean():.3f}", flush=True)


go("register waves", rf.FX, rf.FS)
go("register waves + kept numbers since the largest", np.concatenate([rf.FX, kept_waves(rf.Xf)], -1), np.concatenate([rf.FS, kept_waves(rf.X)], -1))
go("register waves + all kept numbers (no wipe)", np.concatenate([rf.FX, kept_waves(rf.Xf, False)], -1), np.concatenate([rf.FS, kept_waves(rf.X, False)], -1))
go("register waves + every input since the largest", np.concatenate([rf.FX, kept_waves(rf.Xf, True, True)], -1), np.concatenate([rf.FS, kept_waves(rf.X, True, True)], -1))
go("register waves + every input (no wipe)", np.concatenate([rf.FX, kept_waves(rf.Xf, False, True)], -1), np.concatenate([rf.FS, kept_waves(rf.X, False, True)], -1))
