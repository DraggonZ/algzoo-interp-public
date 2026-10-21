"""The two hand-built networks (v5: 98.97%, own algorithm; v18: 96.0%, trained layout) under the same role-free
formulas as gao/sim_across.py. Accuracy only (their raw scores have no common temperature)."""
import numpy as np
from algzoo.handbuilt import build_v5
from algzoo.handbuilt_v18 import build_v18
import simlib
from simlib import lists
from sim_across import feats
Xf, Xs = lists(101, 50_000), lists(202, 50_000); Ff, Fs = feats(Xf), feats(Xs); A = Ff.reshape(-1, Ff.shape[-1])
for name, net in (("v5", build_v5()), ("v18", build_v18())):
    P = tuple(np.asarray(p, float) for p in net.params)
    _, Pf, _ = simlib.run(Xf, params=P); Ls, _, Hs = simlib.run(Xs, params=P); y = Xs.argsort(1)[:, -2]
    sims, r2 = {}, []
    for n in range(16):
        co = np.linalg.lstsq(A, Pf[:, :, n].reshape(-1), rcond=None)[0]; sims[n] = np.maximum(Fs @ co, 0)
        v = Hs[:, :, n].var(); r2.append(1 - ((sims[n] - Hs[:, :, n]) ** 2).mean() / v if v > 1e-9 else 1.0)
    one = [(simlib.run(Xs, {n: sims[n]}, params=P)[0].argmax(1) == y).mean() for n in range(16)]
    al = (simlib.run(Xs, sims, params=P)[0].argmax(1) == y).mean()
    print(f"{name}: acc {(Ls.argmax(1) == y).mean():.3f}, R2 {np.mean(r2):.3f}, one neuron median {np.median(one):.3f} worst {min(one):.3f}, all 16 {al:.3f}")
