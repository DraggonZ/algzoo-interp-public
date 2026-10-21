"""Fit each neuron's formula to its value after the ReLU (minimize mean squared error of relu(F w) against the neuron's
activation; Adam, full batch on CPU, started from least squares), instead of least squares on pre-activations, which
is dominated by deeply-off steps. These are coefficients of a description, not network weights. If the twin then passes
with its own variables in the menu, the earlier failure was a fitting artifact."""
import sys
import numpy as np
import torch
from algzoo.handbuilt_v18 import build_v18
import simlib
from simlib import lists
from sim_across import feats
from twin_rightvars import feats_plus
torch.set_num_threads(4)


def fit(F, h, F_s, steps=1500):
    A = F.reshape(-1, F.shape[-1]); co0 = np.linalg.lstsq(A, h.reshape(-1), rcond=None)[0]
    scale = A.std(0) + 1e-9
    At = torch.tensor(A / scale, dtype=torch.float64); ht = torch.tensor(h.reshape(-1), dtype=torch.float64)
    w = torch.nn.Parameter(torch.tensor(co0 * scale)); opt = torch.optim.Adam([w], lr=1e-2 * float(ht.std() + 1e-9))
    for _ in range(steps):
        loss = ((torch.relu(At @ w) - ht) ** 2).mean(); opt.zero_grad(); loss.backward(); opt.step()
    return np.maximum(F_s @ (w.detach().numpy() / scale), 0)


net, menu = sys.argv[1], sys.argv[2]
P = tuple(np.asarray(p, float) for p in build_v18().params) if net == "v18" else (simlib.WI, simlib.WH, simlib.WO)
fn = feats_plus if menu == "plus" else feats
Xf, Xs = lists(101, 50_000), lists(202, 50_000); y = Xs.argsort(1)[:, -2]; Ff, Fs = fn(Xf), fn(Xs)
_, _, Hf = simlib.run(Xf, params=P); Ls, _, Hs = simlib.run(Xs, params=P); base = (Ls.argmax(1) == y).mean()
sims, one, r2 = {}, [], []
for n in range(16):
    sims[n] = fit(Ff, Hf[:, :, n], Fs); v = Hs[:, :, n].var()
    r2.append(1 - ((sims[n] - Hs[:, :, n]) ** 2).mean() / v if v > 1e-12 else 1.0)
    one.append((simlib.run(Xs, {n: sims[n]}, params=P)[0].argmax(1) == y).mean())
al = (simlib.run(Xs, sims, params=P)[0].argmax(1) == y).mean(); one = np.array(one)
print(f"{net} {menu} menu, fitted after the ReLU: base {base:.3f}; variance matched median {np.median(r2):.4f}; one neuron median {np.median(one):.3f} worst {one.min():.3f}; within 1 point {np.sum(one > base - 0.01)}/16; all 16 {al:.3f}")
print("   one at a time:", " ".join(f"{o:.3f}" for o in one))
