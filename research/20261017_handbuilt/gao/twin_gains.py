"""Is the twin's failure under fitted formulas caused by its very large gains (clock and flag 10,000, detector 1,000)?
Rebuild v18 with moderate gains and score fitted (role-free, gao/sim_across.py) simulation: accuracy with each neuron
replaced alone (median and worst over the 16) and with all 16 replaced. 50,000 lists; fit seed 101, score seed 202."""
import numpy as np
from algzoo.handbuilt_v18 import build_v18
import simlib
from simlib import lists
from sim_across import feats
Xf, Xs = lists(101, 50_000), lists(202, 50_000); Ff, Fs = feats(Xf), feats(Xs); A = Ff.reshape(-1, Ff.shape[-1]); y = Xs.argsort(1)[:, -2]
for g, w, a in ((1e4, 1e4, 1000), (1000, 1000, 100), (100, 100, 100), (100, 100, 20), (30, 30, 20), (10, 10, 10)):
    P = tuple(np.asarray(p, float) for p in build_v18(g=g, w=w, a=a).params)
    _, Pf, _ = simlib.run(Xf, params=P); Ls, _, Hs = simlib.run(Xs, params=P); base = (Ls.argmax(1) == y).mean()
    sims, r2 = {}, []
    for n in range(16):
        co = np.linalg.lstsq(A, Pf[:, :, n].reshape(-1), rcond=None)[0]; sims[n] = np.maximum(Fs @ co, 0)
        v = Hs[:, :, n].var(); r2.append(1 - ((sims[n] - Hs[:, :, n]) ** 2).mean() / v if v > 1e-12 else 1.0)
    one = np.array([(simlib.run(Xs, {n: sims[n]}, params=P)[0].argmax(1) == y).mean() for n in range(16)])
    al = (simlib.run(Xs, sims, params=P)[0].argmax(1) == y).mean()
    print(f"gains clock {g:g}, flag {w:g}, detector {a:g}: accuracy {base:.3f}; variance matched median {np.median(r2):.4f}; "
          f"one neuron replaced: median {np.median(one):.3f}, worst {one.min():.3f}, neurons within 1 point {np.sum(one > base - 0.01)}; all 16 {al:.3f}", flush=True)
