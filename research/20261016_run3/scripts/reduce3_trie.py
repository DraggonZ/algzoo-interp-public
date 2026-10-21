"""Cost of the frozen three-input estimate (reduce3_det.py) when runs that share their first steps share the work.

All of its inputs are the constant background except three positions, so many runs agree on a prefix. A run's state after
t steps depends only on its first t inputs, so each distinct prefix needs one step (d*d + d multiply-adds), and each distinct
full sequence one readout. Two inputs that are equal (ties) also make (p1, p2, p3) and (p1, p3, p2) the same run. This script
re-implements the estimate's runs as one prefix tree, checks that the scores equal the batch runs, and counts the work.
The estimate itself is unchanged.
"""
import itertools
import sys

import numpy as np

sys.path.insert(0, "/data/output/research/algzoo/20261016_run3/scripts")
import reduce3_det as R  # noqa: E402

from algzoo.core import example, zoo  # noqa: E402

T = np.array(list(itertools.permutations(range(10), 3))); n = len(T); ar = np.arange(n)


def inputs(m1, m2, m3, sel=None):
    X = np.full((n, 10), R.B); X[ar, T[:, 0]] = m1; X[ar, T[:, 1]] = m2; X[ar, T[:, 2]] = m3
    return X if sel is None else X[sel]


def tree_scores(params, X):
    """Scores for all rows of X via a prefix tree; returns scores and the number of step and readout evaluations."""
    w_hi, w_hh, w_oh = params
    U, inv = np.unique(X, axis=0, return_inverse=True)
    steps = 0
    keys = np.zeros((len(U), 0)); H = np.zeros((1, len(w_hi))); parent = np.zeros(len(U), dtype=int)
    for t in range(10):
        keys = U[:, : t + 1]
        P, pinv = np.unique(keys, axis=0, return_inverse=True)
        # parent of each new prefix = its first t inputs (index into previous level), via any member row
        first = np.zeros(len(P), dtype=int); first[pinv] = np.arange(len(U))
        Hn = np.maximum(0, H[parent[first]] @ w_hh.T + P[:, t, None] * w_hi)
        steps += len(P)
        parent = pinv; H = Hn
    S = H[parent] @ w_oh.T
    return S[inv.ravel()], steps, len(U)



def shared_cost(params):
    params = tuple(np.asarray(q, np.float64) for q in params)
    d = len(params[0])
    est, ops, det = R.estimate(params)
    # rebuild the estimate's run list (same order and selections as reduce3_det.estimate)
    runs = [inputs(R.L_LO + R.D12, R.L_LO, R.L_LO - R.D23), inputs(R.L_HI + R.D12, R.L_HI, R.L_HI - R.D23)]
    M23 = det["w23"]  # only used for selection masks below; recompute margins for the masks
    S, _ = R.run_batch(params, inputs(R.L_MID + R.D12, R.L_MID, R.L_MID)); m23 = R.margin(S, T[:, 1])[0]
    S, _ = R.run_batch(params, inputs(R.L_MID + R.D12, R.L_MID, R.L_MID - R.EPS if hasattr(R, "EPS") else R.L_MID - 0.02))
    S2, _ = R.run_batch(params, inputs(R.L_MID, R.L_MID, R.L_MID - R.D23)); m12 = R.margin(S2, T[:, 1])[0]
    eps = 0.02
    S, _ = R.run_batch(params, inputs(R.L_MID + R.D12, R.L_MID, R.L_MID - eps)); m23e = R.margin(S, T[:, 1])[0]
    S, _ = R.run_batch(params, inputs(R.L_MID + eps, R.L_MID, R.L_MID - R.D23)); m12e = R.margin(S, T[:, 1])[0]
    need_mid = ((m23 < 0) & (m23e < 0)) | ((m12 < 0) & (m12e < 0))
    runs += [inputs(R.L_MID + R.D12, R.L_MID, R.L_MID - R.D23, need_mid),
             inputs(R.L_MID + R.D12, R.L_MID, R.L_MID), inputs(R.L_MID + R.D12, R.L_MID, R.L_MID - eps, m23 < 0),
             inputs(R.L_MID, R.L_MID, R.L_MID - R.D23), inputs(R.L_MID + eps, R.L_MID, R.L_MID - R.D23, m12 < 0)]
    X = np.concatenate(runs)
    St, steps, leaves = tree_scores(params, X)
    Sb, _ = R.run_batch(params, X)
    assert np.allclose(St, Sb, atol=1e-9), "tree and batch scores differ"
    integral_ops = n * len(R.M2) * 6
    tree_ops = steps * (d * d + d) + leaves * params[2].size + integral_ops
    return est, ops, tree_ops, len(X), leaves, steps

if __name__ == "__main__":
    for name in ["example", 0, 1, 2, 3, 4]:
        est, ops, tree_ops, nr, leaves, steps = shared_cost(example() if name == "example" else zoo(16, 10, name))
        print(f"{name}: estimate {est:.4f}; runs {nr}; distinct inputs {leaves}; prefix-tree steps {steps} (separate runs need {10 * nr}); "
              f"cost with sharing {tree_ops / 2880:.0f} samples' worth (was {ops / 2880:.0f}); scores identical", flush=True)
