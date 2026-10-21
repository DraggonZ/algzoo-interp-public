"""Sanity check for direction_state.py / direction_active.py: with the true previous values of all 16 neurons and the
current input as quantities, a least-squares fit of the cancelling neurons' pre-activations must be exact."""
import numpy as np
import io, contextlib
with contextlib.redirect_stdout(io.StringIO()):
    import register_features as rf
import simlib
from simlib import run

_, _, Hf = simlib.run(rf.Xf); ns = rf.ns


def state_feats(Xs, Hs):
    prev = np.zeros_like(Hs); prev[:, 1:] = Hs[:, :-1]
    return np.concatenate([prev, Xs[..., None]], -1)


Ff, Fs = state_feats(rf.Xf, Hf), state_feats(rf.X, rf.H)
sims = {}
for n in ns:
    co = np.linalg.lstsq(Ff.reshape(-1, 17), rf.PREf[:, :, n].reshape(-1), rcond=None)[0]; sims[n] = np.maximum(Fs @ co, 0)
    print(f"neuron {n}: max |coefficient - true weight| {np.abs(co - np.r_[simlib.WH[n], simlib.WI[n]]).max():.2e}; max |sim - true| {np.abs(sims[n] - rf.H[:, :, n]).max():.2e}")
print("accuracy with these four substituted:", (run(rf.X, sims)[0].argmax(1) == rf.y).mean())
print("check rf.X matches the lists rf.H came from:", np.allclose(simlib.run(rf.X[:100])[2], rf.H[:100]))
