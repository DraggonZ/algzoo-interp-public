"""Blind version of the hidden-signal test: find groups of neurons whose output contributions cancel, without labels.

Each neuron's contribution to the centered scores is c_i = Wc[:, i] * h_i (a 10-vector). G_ij = mean of c_i . c_j over
states. A strongly negative G_ij means neurons i and j push the scores in opposite directions. Split the neurons by the
sign of the leading eigenvector of G (the direction of largest shared push), and report how much the two groups cancel.
States: (a) single-input clock trajectories, (b) random full inputs.
"""
import numpy as np

from algzoo.core import example, zoo

C_TRUE = {1, 2, 4, 6, 7, 8}


def contributions(params, states):
    w_oh = params[2]
    Wc = w_oh - w_oh.mean(0)
    return np.einsum("ji,si->sij", Wc, states)        # [state, neuron, slot]


def states_lone(params):
    w_hi, w_hh, _ = params
    S = []
    for v in (0.5, 1.0, 2.0):
        h = np.maximum(0, v * w_hi)
        for k in range(1, 10):
            h = np.maximum(0, w_hh @ h); S.append(h.copy())
    return np.array(S)


def states_random(params, N=5000):
    w_hi, w_hh, _ = params
    X = np.random.default_rng(0).standard_normal((N, 10))
    h = np.zeros((N, len(w_hi)))
    for t in range(10):
        h = np.maximum(0, h @ w_hh.T + X[:, t, None] * w_hi)
    return h


for name in ["example", 2, 3]:
    p = tuple(np.asarray(q, np.float64) for q in (example() if name == "example" else zoo(16, 10, name)))
    for lab, S in [("single-input clocks", states_lone(p)), ("random full inputs", states_random(p))]:
        c = contributions(p, S)
        G = np.einsum("sik,sjk->ij", c, c) / len(S)
        ev, V = np.linalg.eigh(G)
        v = V[:, -1]
        A = [i for i in range(16) if v[i] > 0]; B = [i for i in range(16) if v[i] <= 0]
        sa = c[:, A].sum(1); sb = c[:, B].sum(1)
        canc = np.mean(np.linalg.norm(sa + sb, axis=1)) / np.mean(np.linalg.norm(sa, axis=1) + np.linalg.norm(sb, axis=1)) * 2
        corr = np.mean([np.corrcoef(sa[s], sb[s])[0, 1] for s in range(0, len(S), max(1, len(S) // 500))])
        overlap = max(len(set(A) & C_TRUE) + len(set(B) - C_TRUE), len(set(B) & C_TRUE) + len(set(A) - C_TRUE))
        print(f"{name} / {lab}: groups {A} | {B}; mean corr {corr:+.2f}; size of sum relative to parts {canc:.2f}; "
              f"agreement with comparing/other split {overlap}/16" + ("" if name == "example" else " (split labels are ARC-model numbering, only meaningful for the example)"))
