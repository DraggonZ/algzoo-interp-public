"""The cancelling neurons as their own clipped rotation with a fitted drive. Model: c(t) = ReLU(W_cc c(t-1) + B q(t)),
W_cc the trained weights among neurons 1, 6, 7, 8, q(t) input quantities (register waves, register_features.py, plus
the input), B fitted by least squares to the drive the four actually receive from the rest of the network and the
input (pre-activation minus W_cc c_true(t-1)), on steps where the neuron is on. Then run the model freely from the
inputs alone (its own clipping history included) and substitute the four. Compared with plain formulas."""
import numpy as np
import io, contextlib
with contextlib.redirect_stdout(io.StringIO()):
    import register_features as rf
import simlib
from simlib import run

C = [1, 6, 7, 8]; Wcc = simlib.WH[np.ix_(C, C)]
_, PREf, Hf = simlib.run(rf.Xf)
prevf = np.zeros_like(Hf[:, :, C]); prevf[:, 1:] = Hf[:, :-1][:, :, C]
drive = PREf[:, :, C] - prevf @ Wcc.T                                        # what the rest of the network supplies
def recurrent(Qf, Qs, lab):
    Bm = []
    for i, n in enumerate(C):
        m = Hf[:, :, n].reshape(-1) > 0
        Bm.append(np.linalg.lstsq(Qf.reshape(-1, Qf.shape[-1])[m], drive[:, :, i].reshape(-1)[m], rcond=None)[0])
    Bm = np.stack(Bm, 0); c = np.zeros((len(Qs), 4)); out = np.zeros((len(Qs), 10, 4))
    for t in range(10):
        c = np.maximum(c @ Wcc.T + Qs[:, t] @ Bm.T, 0); out[:, t] = c
    V = rf.H[:, :, C]; mm = (V > 0).all(-1)
    cang = (out * V).sum(-1)[mm] / np.maximum(np.linalg.norm(out, axis=-1)[mm] * np.linalg.norm(V, axis=-1)[mm], 1e-9)
    acc = (run(rf.X, {n: out[:, :, i] for i, n in enumerate(C)})[0].argmax(1) == rf.y).mean()
    print(f"own clipped rotation, drive from {lab:28s} ({Qf.shape[-1]:2d}): angle median {np.degrees(np.arccos(np.clip(np.median(cang), -1, 1))):.2f} deg; accuracy {acc:.3f}", flush=True)


from sim_across import feats as menu
from sim_groups import account_features
recurrent(np.concatenate([rf.FX, rf.Xf[..., None]], -1), np.concatenate([rf.FS, rf.X[..., None]], -1), "register waves + input")
recurrent(menu(rf.Xf), menu(rf.X), "general menu")
recurrent(account_features(rf.Xf, 1), account_features(rf.X, 1), "our account's quantities")
recurrent(rf.Xf[..., None], rf.X[..., None], "the input alone")

plain = {}
for n in C:
    m = Hf[:, :, n].reshape(-1) > 0
    co = np.linalg.lstsq(rf.FX.reshape(-1, rf.FX.shape[-1])[m], rf.PREf[:, :, n].reshape(-1)[m], rcond=None)[0]; plain[n] = np.maximum(rf.FS @ co, 0)
print(f"plain register-wave formulas:           accuracy {(run(rf.X, plain)[0].argmax(1) == rf.y).mean():.3f}")
