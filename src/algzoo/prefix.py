"""Evaluate many structured inputs, sharing the work of common first steps.

The RNN's state after t steps depends only on the first t inputs, so inputs that agree on a prefix can share its
computation. Chosen inputs (for example a constant background with a few spikes) share prefixes often; random inputs
almost never do. This is the cost model behind run 3's recount of its sampling-free estimate.
"""
import numpy as np


def tree_scores(params, X):
    """Scores for every row of X [N, n]. Returns (scores [N, n_out], steps, distinct): steps is the number of distinct
    prefixes evaluated (one RNN step each), distinct the number of distinct full inputs (one readout each)."""
    w_hi, w_hh, w_oh = (np.asarray(p, np.float64) for p in params)
    U, inv = np.unique(X, axis=0, return_inverse=True)
    H = np.zeros((1, len(w_hi)))
    parent = np.zeros(len(U), dtype=int)
    steps = 0
    for t in range(X.shape[1]):
        P, pinv = np.unique(U[:, : t + 1], axis=0, return_inverse=True)
        pinv = pinv.ravel()
        first = np.zeros(len(P), dtype=int)
        first[pinv] = np.arange(len(U))
        H = np.maximum(0, H[parent[first]] @ w_hh.T + P[:, t, None] * w_hi)
        steps += len(P)
        parent = pinv
    return (H[parent] @ w_oh.T)[inv.ravel()], steps, len(U)


def cost_in_runs(params, steps, distinct, seq_len=10):
    """Multiply-adds of a prefix-tree evaluation divided by those of one full run of seq_len steps."""
    w_hi, _, w_oh = params
    d = len(w_hi)
    return (steps * (d * d + d) + distinct * np.size(w_oh)) / (seq_len * (d * d + d) + np.size(w_oh))
