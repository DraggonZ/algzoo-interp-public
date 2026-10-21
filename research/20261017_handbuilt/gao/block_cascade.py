"""Cascade of blocks: the memory as its own recurrent system whose drive is the trained cross-weights applied to the
other groups' values (cancelling neurons from their own block account; store, detector and gates from plain formulas)
plus the input, instead of a fitted formula. Variants for diagnosis: other groups exact; cancellers exact and the rest
simulated; all simulated. Substitute the memory only, then memory and cancellers. General menu as quantities."""
import numpy as np
import simlib
from block_recurrent import Xs, ys, block, plain, G, acc

M = G["memory"]; Cn = G["cancelling"]; O = [n for n in range(16) if n not in M]
_, _, Hs = simlib.run(Xs)
Wmm = simlib.WH[np.ix_(M, M)]; Wmo = simlib.WH[np.ix_(M, O)]


def memory_from(others):
    c = np.zeros((len(Xs), len(M))); out = np.zeros((len(Xs), 10, len(M)))
    for t in range(10):
        drive = (np.stack([others[n][:, t - 1] for n in O], -1) @ Wmo.T) if t else 0
        c = np.maximum(c @ Wmm.T + drive + Xs[:, t:t + 1] * simlib.WI[M], 0); out[:, t] = c
    return {n: out[:, :, i] for i, n in enumerate(M)}


true = {n: Hs[:, :, n] for n in range(16)}
canc = block(Cn); rest = plain(G["store+detector"] + G["gates"])
for lab, others in (("others exact", true), ("cancellers from their block, rest exact", {**true, **canc}),
                    ("cancellers exact, store/detector/gates from formulas", {**true, **rest}),
                    ("all others simulated", {**true, **canc, **rest})):
    mem = memory_from(others)
    print(f"memory from the weights, drive {lab:52s}: memory replaced {acc(mem):.3f}; memory and cancellers replaced {acc({**mem, **canc}):.3f}", flush=True)

sd = plain(G["store+detector"]); gt = plain(G["gates"]); g5 = plain([5]); g9 = plain([9])
for lab, others in (("store/detector from formulas, gates exact", {**true, **canc, **sd}),
                    ("gates from formulas, store/detector exact", {**true, **canc, **gt}),
                    ("keep gate 5 from its formula, rest exact", {**true, **canc, **g5}),
                    ("late gate 9 from its formula, rest exact", {**true, **canc, **g9})):
    mem = memory_from(others)
    print(f"memory from the weights, drive {lab:52s}: memory replaced {acc(mem):.3f}", flush=True)

from sim_groups import account_features
from block_recurrent import Xf, Hf, PREf, Qf, Qs


def fit_with(n, Ff, Fs):
    m = Hf[:, :, n].reshape(-1) > 0
    co = np.linalg.lstsq(Ff.reshape(-1, Ff.shape[-1])[m], PREf[:, :, n].reshape(-1)[m], rcond=None)[0]; return np.maximum(Fs @ co, 0)


for n in (9, 5):
    for lab, (Ff, Fs) in (("account quantities", (account_features(Xf, n), account_features(Xs, n))),
                          ("menu", (Qf, Qs)), ("account + menu", (np.concatenate([account_features(Xf, n), Qf], -1), np.concatenate([account_features(Xs, n), Qs], -1)))):
        g = {n: fit_with(n, Ff, Fs)}; mem = memory_from({**true, **canc, **g})
        err = np.abs(g[n] - Hs[:, :, n]); act = Hs[:, :, n] > 0
        print(f"gate {n} from {lab:20s}: its median error when on {np.median(err[act]):.3f} (typical value {np.median(Hs[:, :, n][act]):.2f}); "
              f"memory driven by it, rest exact: {acc(mem):.3f}; gate alone replaced: {acc(g):.3f}", flush=True)
