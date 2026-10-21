"""Where does the account's simulation lose the output? Per neuron, substituting at every step versus only at the
last step (then the error reaches the scores only through the readout, not through later steps)."""
import numpy as np
from simlib import lists, run, score, fit_relu
from sim_groups import account_features

Xf, Xs = lists(101), lists(202)
_, Pf, Hf = run(Xf); Ls, Ps, Hs = run(Xs)
print(f"{'neuron':>6s} {'R2':>6s} {'rms err':>7s} {'all steps acc':>13s} {'KL':>6s} {'last step acc':>13s} {'KL':>6s} {'readout norm':>12s}")
for n in range(16):
    sim = fit_relu(account_features(Xf, n), Pf[:, :, n], account_features(Xs, n))[0]
    r2 = 1 - ((Hs[:, :, n] - sim) ** 2).mean() / Hs[:, :, n].var()
    rms = np.sqrt(((Hs[:, :, n] - sim) ** 2).mean())
    a = score(Xs, Ls, run(Xs, {n: sim})[0])
    last = Hs[:, :, n].copy(); last[:, 9] = sim[:, 9]
    b = score(Xs, Ls, run(Xs, {n: last})[0])
    print(f"{n:6d} {r2:6.3f} {rms:7.3f} {a['acc']:13.3f} {a['kl']:6.2f} {b['acc']:13.3f} {b['kl']:6.2f} {np.linalg.norm(Hs[0,0,:0]) if False else np.linalg.norm(__import__('simlib').WO[:, n]):12.2f}")
