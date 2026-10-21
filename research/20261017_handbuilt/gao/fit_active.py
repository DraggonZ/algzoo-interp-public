"""ReLU regression by active-set iteration: start with least squares on steps where the neuron is on; then repeatedly
refit on steps where the neuron is on OR the current formula predicts it on, with the true activation (zero when off)
as target. This fits the formula to the neuron's value after the ReLU, without the deeply-off steps that distort least
squares on pre-activations. Applied to v18 (menu with its own variables, then the role-free menu) and to the trained
network (role-free menu). Coefficients of a description, not network weights."""
import sys
import numpy as np
from algzoo.handbuilt_v18 import build_v18
import simlib
from simlib import lists
from sim_across import feats
from twin_rightvars import feats_plus


def fit_active(F, h, iters=15):
    A = F.reshape(-1, F.shape[-1]); t = h.reshape(-1); on = t > 0
    if on.sum() <= A.shape[1]:
        return np.zeros(A.shape[1])
    co = np.linalg.lstsq(A[on], t[on], rcond=None)[0]
    for _ in range(iters):
        act = on | (A @ co > 0); co_new = np.linalg.lstsq(A[act], t[act], rcond=None)[0]
        if np.allclose(co_new, co, rtol=1e-9, atol=1e-12):
            break
        co = co_new
    return co


def evaluate(P, fn, Xf, Xs):
    y = Xs.argsort(1)[:, -2]; Ff, Fs = fn(Xf), fn(Xs)
    _, _, Hf = simlib.run(Xf, params=P); Ls, _, Hs = simlib.run(Xs, params=P); base = (Ls.argmax(1) == y).mean()
    sims, one, r2 = {}, [], []
    for n in range(16):
        co = fit_active(Ff, Hf[:, :, n]); sims[n] = np.maximum(Fs @ co, 0); v = Hs[:, :, n].var()
        r2.append(1 - ((sims[n] - Hs[:, :, n]) ** 2).mean() / v if v > 1e-12 else 1.0)
        one.append((simlib.run(Xs, {n: sims[n]}, params=P)[0].argmax(1) == y).mean())
    al = (simlib.run(Xs, sims, params=P)[0].argmax(1) == y).mean()
    return base, np.array(r2), np.array(one), al


if __name__ == "__main__":
    Xf, Xs = lists(101, 50_000), lists(202, 50_000)
    v18 = tuple(np.asarray(p, float) for p in build_v18().params); tr = (simlib.WI, simlib.WH, simlib.WO)
    for label, P, fn in (("v18, menu with its own variables", v18, feats_plus), ("v18, role-free menu", v18, feats),
                         ("trained, role-free menu", tr, feats), ("trained, menu plus v18's variables", tr, feats_plus)):
        base, r2, one, al = evaluate(P, fn, Xf, Xs)
        print(f"{label:36s}: base {base:.3f}; variance matched median {np.median(r2):.4f}; one neuron median {np.median(one):.3f} worst {one.min():.3f}; within 1 point {np.sum(one > base - 0.01):2d}/16; all 16 {al:.3f}", flush=True)
