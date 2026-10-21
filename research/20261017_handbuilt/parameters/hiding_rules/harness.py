"""Shared harness for the hiding-neuron rules (README experiment 21). Metric as gao/sim_tolerance.py: fit the formula's
coefficients by least squares on pre-activations (all steps pooled, lists of seed 101), then ReLU, and take the rms error
against the true activation over all lists and steps of seed 202 (200,000 lists each). Also the accuracy and KL from
ARC's output when the neuron is replaced by its formula (the rest of the network real)."""
import sys
import numpy as np
sys.path.insert(0, "/data/output/research/algzoo/20261017_handbuilt/gao")
import simlib  # noqa: E402
from simlib import lists, run, task_vars, score  # noqa: E402

HID = [1, 6, 7, 8]
BASE_RMS = {1: 1.846, 6: 1.968, 7: 0.628, 8: 2.133}   # gao/sim_tolerance.log, the account's formulas
Xf, Xs = lists(101), lists(202)
_, Pf, Hf = run(Xf)
Ls, Ps, Hs = run(Xs)


def lag(a, k=1):
    out = np.zeros_like(a); out[:, k:] = a[:, :-k]; return out


def evaluate(feat_fn, neurons=HID, verbose=True, weights=None):
    """feat_fn(X, n) -> (B, 10, nf). Returns {n: (rms, acc, kl, coef)}."""
    out = {}
    for n in neurons:
        Ff, Fs = feat_fn(Xf, n), feat_fn(Xs, n)
        A = Ff.reshape(-1, Ff.shape[-1]); yv = Pf[:, :, n].reshape(-1)
        co = np.linalg.lstsq(A, yv, rcond=None)[0]
        sim = np.maximum(Fs @ co, 0)
        rms = np.sqrt(((sim - Hs[:, :, n]) ** 2).mean())
        sc = score(Xs, Ls, run(Xs, {n: sim})[0])
        out[n] = (rms, sc["acc"], sc["kl"], co)
        if verbose:
            print(f"  neuron {n}: {Ff.shape[-1]:3d} constants, rms {rms:.3f} ({BASE_RMS[n] / rms:.2f}x better than the account), "
                  f"replaced: accuracy {sc['acc']:.4f}, KL {sc['kl']:.4f}", flush=True)
    return out
