"""An account of the cancelling neurons as their own small clipped rotation. Simulate neurons 1, 6, 7, 8 by running
c(t) = ReLU(W_cc c(t-1) + W_co o(t-1) + w_c x(t)) with the trained weights, where o are the other 12 neurons given by
formulas of the input (least squares on active steps): either the register waves, or the general menu. Substitute the
four simulated neurons (only them) and compare with plain formulas for the four. Also the drivers given exactly
(o = true values), which reproduces the network and checks the code."""
import numpy as np
import io, contextlib
with contextlib.redirect_stdout(io.StringIO()):
    import register_features as rf
import simlib
from simlib import run
from sim_across import feats as menu

C = [1, 6, 7, 8]; O = [n for n in range(16) if n not in C]
_, _, Hf = simlib.run(rf.Xf)


def formulas(Ff, Fs, neurons):
    out = {}
    for n in neurons:
        m = Hf[:, :, n].reshape(-1) > 0
        if m.sum() < 10:
            out[n] = np.zeros(Fs.shape[:2]); continue
        co = np.linalg.lstsq(Ff.reshape(-1, Ff.shape[-1])[m], rf.PREf[:, :, n].reshape(-1)[m], rcond=None)[0]; out[n] = np.maximum(Fs @ co, 0)
    return out


def sub(o):
    B = len(rf.X); c = np.zeros((B, 4)); out = np.zeros((B, 10, 4))
    for t in range(10):
        prev = np.zeros(16)
        drive = (np.stack([o[n][:, t - 1] for n in O], -1) @ simlib.WH[np.ix_(C, O)].T) if t else 0
        c = np.maximum(c @ simlib.WH[np.ix_(C, C)].T + drive + rf.X[:, t:t + 1] * simlib.WI[C], 0); out[:, t] = c
    return {n: out[:, :, i] for i, n in enumerate(C)}


def acc(sims):
    return (run(rf.X, sims)[0].argmax(1) == rf.y).mean()


MX, MS = menu(rf.Xf), menu(rf.X)
print(f"plain register-wave formulas for the four: {acc(formulas(rf.FX, rf.FS, C)):.3f}")
print(f"plain menu formulas for the four:          {acc(formulas(MX, MS, C)):.3f}")
print(f"own clipped rotation, drivers exact:        {acc(sub({n: rf.H[:, :, n] for n in O})):.3f}")
for lab, (Ff, Fs) in (("register waves", (rf.FX, rf.FS)), ("general menu", (MX, MS)), ("both", (np.concatenate([rf.FX, MX], -1), np.concatenate([rf.FS, MS], -1)))):
    print(f"own clipped rotation, drivers from {lab:15s}: {acc(sub(formulas(Ff, Fs, O))):.3f}", flush=True)
