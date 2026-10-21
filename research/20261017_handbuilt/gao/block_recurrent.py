"""Block accounts: each group of neurons as its own small recurrent system (the trained weights within the group, with
ReLU), driven by a formula of input quantities fitted by least squares to the drive it actually receives from outside
the group (pre-activation minus the within-group part, true previous values, steps where the neuron is on). Run freely
from the inputs; substitute one group at a time, then several. Quantities: the general 30-quantity menu."""
import numpy as np
import simlib
from simlib import lists, run
from sim_across import feats as menu

Xf, Xs = lists(3030, 100_000), lists(3131, 50_000); ys = Xs.argsort(1)[:, -2]
_, PREf, Hf = simlib.run(Xf); Qf, Qs = menu(Xf), menu(Xs)
G = {"cancelling": [1, 6, 7, 8], "memory": [0, 3, 10, 11, 12, 13, 14, 15], "store+detector": [2, 4], "gates": [5, 9]}


def block(ns):
    W = simlib.WH[np.ix_(ns, ns)]; prev = np.zeros_like(Hf[:, :, ns]); prev[:, 1:] = Hf[:, :-1][:, :, ns]
    drive = PREf[:, :, ns] - prev @ W.T; Bm = []
    for i, n in enumerate(ns):
        m = Hf[:, :, n].reshape(-1) > 0
        Bm.append(np.linalg.lstsq(Qf.reshape(-1, Qf.shape[-1])[m], drive[:, :, i].reshape(-1)[m], rcond=None)[0] if m.sum() > 50 else np.zeros(Qf.shape[-1]))
    Bm = np.stack(Bm, 0); c = np.zeros((len(Xs), len(ns))); out = np.zeros((len(Xs), 10, len(ns)))
    for t in range(10):
        c = np.maximum(c @ W.T + Qs[:, t] @ Bm.T, 0); out[:, t] = c
    return {n: out[:, :, i] for i, n in enumerate(ns)}


def plain(ns):
    sims = {}
    for n in ns:
        m = Hf[:, :, n].reshape(-1) > 0
        co = np.linalg.lstsq(Qf.reshape(-1, Qf.shape[-1])[m], PREf[:, :, n].reshape(-1)[m], rcond=None)[0]; sims[n] = np.maximum(Qs @ co, 0)
    return sims


acc = lambda sims: (run(Xs, sims)[0].argmax(1) == ys).mean()
print(f"network accuracy {acc({}):.3f}")
for g, ns in G.items():
    print(f"{g:15s}: plain formulas {acc(plain(ns)):.3f}; own recurrence with fitted drive {acc(block(ns)):.3f}", flush=True)
both = {**block(G["cancelling"]), **block(G["memory"])}
print(f"cancelling and memory together as blocks: {acc(both):.3f}")
allb = {}
for ns in G.values():
    allb.update(block(ns))
print(f"all four groups as blocks (all 16 replaced): {acc(allb):.3f}; all 16 plain formulas: {acc(plain(list(range(16)))):.3f}")
