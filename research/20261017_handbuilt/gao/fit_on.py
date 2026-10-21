"""Fitting artifact check. Least squares on pre-activations at every step is dominated by steps where the neuron is
deeply off (for v18's clock, about -10,000 x the jump), which do not matter after the ReLU. Refit each neuron only on
steps where it is on (pre-activation > 0), then apply the ReLU. Compare with fitting everywhere, for v18 (with its own
variables added to the menu) and for ARC's trained network (role-free menu), and for both with the role-free menu."""
import numpy as np
from algzoo.handbuilt_v18 import build_v18
import simlib
from simlib import lists
from sim_across import feats
from twin_rightvars import feats_plus

Xf, Xs = lists(101, 50_000), lists(202, 50_000); y = Xs.argsort(1)[:, -2]
for net, P in (("v18", tuple(np.asarray(p, float) for p in build_v18().params)), ("trained", (simlib.WI, simlib.WH, simlib.WO))):
    _, Pf, _ = simlib.run(Xf, params=P); Ls, _, Hs = simlib.run(Xs, params=P); base = (Ls.argmax(1) == y).mean()
    for mname, fn in (("role-free menu", feats), ("menu plus v18's variables", feats_plus)):
        Ff, Fs = fn(Xf), fn(Xs)
        for how in ("everywhere", "on-steps only"):
            sims, one = {}, []
            for n in range(16):
                A = Ff.reshape(-1, Ff.shape[-1]); t = Pf[:, :, n].reshape(-1); m = t > 0 if how == "on-steps only" else np.ones(len(t), bool)
                co = np.linalg.lstsq(A[m], t[m], rcond=None)[0] if m.sum() > Ff.shape[-1] else np.zeros(Ff.shape[-1])
                sims[n] = np.maximum(Fs @ co, 0); one.append((simlib.run(Xs, {n: sims[n]}, params=P)[0].argmax(1) == y).mean())
            al = (simlib.run(Xs, sims, params=P)[0].argmax(1) == y).mean(); one = np.array(one)
            print(f"{net:7s} {mname:26s} fit {how:13s}: base {base:.3f}; one neuron median {np.median(one):.3f} worst {one.min():.3f}; within 1 point {np.sum(one > base - 0.01):2d}/16; all 16 {al:.3f}", flush=True)
