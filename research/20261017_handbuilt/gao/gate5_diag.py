"""Keep gate (neuron 5): what does its formula miss? Its pre-activation is exactly W_5 . h(t-1) + w_5 x(t). Per
neuron m, the share of the gate's pre-activation variance (active steps) carried by W_5m h_m(t-1); then formulas from
the account's quantities plus the true previous values of chosen neurons (diagnostic), and the block account's
accuracy (block_full.py) with the keep gate from each."""
import numpy as np
import io, contextlib
with contextlib.redirect_stdout(io.StringIO()):
    from block_cascade import Xs, Hs, true, canc, plain, G, acc, memory_from, fit_with, account_features, Xf, Qf, Qs
import simlib

_, PREf, Hf = simlib.run(Xf)
prev = np.zeros_like(Hf); prev[:, 1:] = Hf[:, :-1]; act = Hf[:, :, 5] > 0
contrib = prev * simlib.WH[5][None, None, :]
tot = PREf[:, :, 5][act].var()
print("share of the keep gate's pre-activation variance by source neuron (active steps):")
print("  " + " ".join(f"{m}:{contrib[:, :, m][act].var() / tot:.2f}" for m in range(16) if contrib[:, :, m][act].var() / tot > 0.01), f"input:{(Xf * simlib.WI[5])[act].var() / tot:.2f}")
prevs = np.zeros_like(Hs); prevs[:, 1:] = Hs[:, :-1]
sd = plain(G["store+detector"]); g9 = {9: fit_with(9, account_features(Xf, 9), account_features(Xs, 9))}
A5f, A5s = np.concatenate([account_features(Xf, 5), Qf], -1), np.concatenate([account_features(Xs, 5), Qs], -1)
for lab, extra in (("none", []), ("store 2", [2]), ("second-largest store 8", [8]), ("cancelling 1, 6, 7, 8", [1, 6, 7, 8]), ("memory", [0, 3, 10, 11, 12, 13, 14, 15])):
    Ff = np.concatenate([A5f, prev[:, :, extra]], -1); Fs = np.concatenate([A5s, prevs[:, :, extra]], -1)
    g5 = {5: fit_with(5, Ff, Fs)}
    mem = memory_from({**true, **canc, **sd, **g5, **g9})
    e = np.abs(g5[5] - Hs[:, :, 5])[Hs[:, :, 5] > 0]
    print(f"keep gate from account+menu + true previous {lab:24s}: median error when on {np.median(e):.3f}; block account, all 16 replaced: {acc({**canc, **sd, **g5, **g9, **mem}):.3f}", flush=True)
