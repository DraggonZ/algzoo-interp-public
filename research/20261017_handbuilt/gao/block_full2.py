"""Block account, version 2: the keep gate (5) runs inside the memory's recurrent block (trained weights among memory
and gate, driven through the cross-weights by the simulated other neurons and the input). Others: cancellers as their
own block with fitted drive; store and detector from menu formulas; late gate from the account's quantities.
All 16 replaced. Variant: the detector also inside the block."""
import numpy as np
import io, contextlib
with contextlib.redirect_stdout(io.StringIO()):
    from block_cascade import Xs, Hs, true, canc, plain, G, acc, fit_with, account_features, Xf
import simlib


def run_block(B, others):
    O = [n for n in range(16) if n not in B]; Wbb = simlib.WH[np.ix_(B, B)]; Wbo = simlib.WH[np.ix_(B, O)]
    c = np.zeros((len(Xs), len(B))); out = np.zeros((len(Xs), 10, len(B)))
    for t in range(10):
        drive = (np.stack([others[n][:, t - 1] for n in O], -1) @ Wbo.T) if t else 0
        c = np.maximum(c @ Wbb.T + drive + Xs[:, t:t + 1] * simlib.WI[B], 0); out[:, t] = c
    return {n: out[:, :, i] for i, n in enumerate(B)}


g9 = {9: fit_with(9, account_features(Xf, 9), account_features(Xs, 9))}
mem = G["memory"]
for lab, B, form in (("memory + keep gate as one block; store, detector formulas", mem + [5], G["store+detector"]),
                     ("memory + keep gate + detector as one block; store formula", mem + [5, 4], [2])):
    f = plain(form); sims = {**canc, **f, **g9}
    blk = run_block(B, {**true, **sims})
    print(f"{lab:62s}: all 16 replaced {acc({**sims, **blk}):.3f}", flush=True)


def run_block_partial_relu(B, others, relu):
    O = [n for n in range(16) if n not in B]; Wbb = simlib.WH[np.ix_(B, B)]; Wbo = simlib.WH[np.ix_(B, O)]
    mask = np.array([n in relu for n in B]); c = np.zeros((len(Xs), len(B))); out = np.zeros((len(Xs), 10, len(B)))
    for t in range(10):
        drive = (np.stack([others[n][:, t - 1] for n in O], -1) @ Wbo.T) if t else 0
        pre = c @ Wbb.T + drive + Xs[:, t:t + 1] * simlib.WI[B]
        c = np.where(mask, np.maximum(pre, 0), pre); out[:, t] = c
    return {n: out[:, :, i] for i, n in enumerate(B)}


f = plain(G["store+detector"]); sims = {**canc, **f, **g9}
for lab, relu in (("all ReLUs kept", mem + [5]), ("memory linear except 0, 15; gate ReLU", [0, 15, 5]), ("memory fully linear; gate ReLU", [5])):
    blk = run_block_partial_relu(mem + [5], {**true, **sims}, relu)
    out = {**sims, **blk}
    out = {n: np.maximum(v, 0) for n, v in out.items()}  # values passed to the real network are non-negative
    print(f"memory + keep gate block, {lab:40s}: all 16 replaced {acc(out):.3f}", flush=True)

from simlib import task_vars
age = task_vars(Xs)["age"]; anypos = np.maximum.accumulate(Xs > 0, axis=1)


def run_block_rule(B, others):
    O = [n for n in range(16) if n not in B]; Wbb = simlib.WH[np.ix_(B, B)]; Wbo = simlib.WH[np.ix_(B, O)]
    i0, i15, i5 = B.index(0), B.index(15), B.index(5); c = np.zeros((len(Xs), len(B))); out = np.zeros((len(Xs), 10, len(B)))
    for t in range(10):
        drive = (np.stack([others[n][:, t - 1] for n in O], -1) @ Wbo.T) if t else 0
        c = c @ Wbb.T + drive + Xs[:, t:t + 1] * simlib.WI[B]
        c[:, i15] = np.maximum(c[:, i15], 0); c[:, i5] = np.maximum(c[:, i5], 0)
        c[:, i0] = np.where((age[:, t] == 2) & anypos[:, t], 0.0, c[:, i0])   # neuron 0's rule: off two steps after a new largest
        out[:, t] = c
    return {n: out[:, :, i] for i, n in enumerate(B)}


blk = run_block_rule(mem + [5], {**true, **sims}); out = {n: np.maximum(v, 0) for n, v in {**sims, **blk}.items()}
print(f"memory + keep gate block, linear except neuron 0 by its rule, 15 and gate ReLU: all 16 replaced {acc(out):.3f}", flush=True)

# Neuron 15: predict its on/off from the sign of a formula of input quantities (register waves + menu), fitted to its
# pre-activation on all steps; use that as its switching rule inside the linear memory.
import io as _io, contextlib as _cl
with _cl.redirect_stdout(_io.StringIO()):
    import register_features as rf
from sim_across import feats as menu
_, PREf2, _ = simlib.run(Xf)
Ff = np.concatenate([menu(Xf), np.concatenate([rf.feats(Xf), Xf[..., None]], -1)], -1)
Fs = np.concatenate([menu(Xs), np.concatenate([rf.feats(Xs), Xs[..., None]], -1)], -1)
co = np.linalg.lstsq(Ff.reshape(-1, Ff.shape[-1]), PREf2[:, :, 15].reshape(-1), rcond=None)[0]
on15 = (Fs @ co) > 0; true_on = Hs[:, :, 15] > 0
print(f"neuron 15 on/off predicted from a formula's sign: agrees on {np.mean(on15 == true_on):.3f} of steps (always-on would agree on {true_on.mean():.3f})")


def run_block_rule15(B, others):
    O = [n for n in range(16) if n not in B]; Wbb = simlib.WH[np.ix_(B, B)]; Wbo = simlib.WH[np.ix_(B, O)]
    i0, i15, i5 = B.index(0), B.index(15), B.index(5); c = np.zeros((len(Xs), len(B))); out = np.zeros((len(Xs), 10, len(B)))
    for t in range(10):
        drive = (np.stack([others[n][:, t - 1] for n in O], -1) @ Wbo.T) if t else 0
        c = c @ Wbb.T + drive + Xs[:, t:t + 1] * simlib.WI[B]
        c[:, i5] = np.maximum(c[:, i5], 0)
        c[:, i0] = np.where((age[:, t] == 2) & anypos[:, t], 0.0, c[:, i0])
        c[:, i15] = np.where(on15[:, t], c[:, i15], 0.0)
        out[:, t] = c
    return {n: out[:, :, i] for i, n in enumerate(B)}


blk = run_block_rule15(mem + [5], {**true, **sims}); out = {n: np.maximum(v, 0) for n, v in {**sims, **blk}.items()}
print(f"memory linear, neuron 0 by its rule, neuron 15 by its formula's sign, gate ReLU: all 16 replaced {acc(out):.3f}", flush=True)
