"""Null copies of ARC's weights: sizes kept, structure lost.

kind "arc":   ARC itself (for checking the wrapper).
kind "block": permute within each of the v2 scorer's 24 source -> target blocks.
kind "row":   permute within each (target row, source group) segment; input weights untouched (one per neuron).

install(kind, seed) replaces algzoo.core.example by a function returning the null's weights. Call it before importing
the scorer or any rule (they bind `example` at import)."""
import numpy as np
import algzoo.core as core

GROUPS = {"store/det": [2, 4], "gates": [5, 9], "hiding": [1, 6, 7, 8], "memory": [0, 3, 10, 11, 12, 13, 14, 15]}
GOF = {n: g for g, ns in GROUPS.items() for n in ns}
_ARC = core.example


def labels():
    """Per flat weight (scorer order: 16 input, WH[j, i] at 16 + 16 j + i, WO[k, i] at 272 + 16 k + i):
    (block label, row label)."""
    blk, row = [], []
    for j in range(16):
        blk.append(f"input -> {GOF[j]}"); row.append(f"in{j}:input")
    for j in range(16):
        for i in range(16):
            blk.append(f"{GOF[i]} -> {GOF[j]}"); row.append(f"in{j}:{GOF[i]}")
    for k in range(10):
        for i in range(16):
            blk.append(f"{GOF[i]} -> readout"); row.append(f"out{k}:{GOF[i]}")
    return np.array(blk), np.array(row)


BLOCK, ROW = labels()


def arc_flat():
    WI, WH, WO = _ARC()
    return np.concatenate([WI.ravel(), WH.ravel(), WO.ravel()])


def null_flat(kind, seed):
    w = arc_flat()
    if kind == "arc":
        return w
    lab = {"block": BLOCK, "row": ROW}[kind]
    rng = np.random.default_rng(10_000 * {"block": 1, "row": 2}[kind] + seed)
    out = w.copy()
    for b in sorted(set(lab)):
        ii = np.flatnonzero(lab == b)
        out[ii] = w[rng.permutation(ii)]
    return out


def null_weights(kind, seed):
    w = null_flat(kind, seed)
    WI, WH, WO = _ARC()
    return (w[:16].reshape(WI.shape).astype(WI.dtype), w[16:272].reshape(16, 16).astype(WH.dtype),
            w[272:].reshape(10, 16).astype(WO.dtype))


_solve = np.linalg.solve
SINGULAR = []


def _safe_solve(a, b):
    """np.linalg.solve, or the minimum-norm least-squares solution where the matrix is singular (a neuron that never
    fires on a null leaves a zero row and column in the state Gram matrix; ARC never reaches this branch)."""
    try:
        return _solve(a, b)
    except np.linalg.LinAlgError:
        SINGULAR.append(np.shape(a))
        return np.linalg.lstsq(a, b, rcond=None)[0]


def install(kind, seed):
    P = null_weights(kind, seed)
    core.example = lambda: tuple(p.copy() for p in P)
    if kind != "arc":
        np.linalg.solve = _safe_solve
    return P


def install_flat(w):
    """Install given flat weights (432, scorer order) as the network."""
    WI, WH, WO = _ARC()
    P = (w[:16].reshape(WI.shape).astype(WI.dtype), w[16:272].reshape(16, 16).astype(WH.dtype),
         w[272:].reshape(10, 16).astype(WO.dtype))
    core.example = lambda: tuple(p.copy() for p in P)
    np.linalg.solve = _safe_solve
    return P


def residual_null(pred, covered_idx, rows_idx, seed, pool=True):
    """ARC with the residuals of a rule shuffled in some rows.
    pred: the rule's predicted values at covered_idx (flat indices). rows_idx: flat indices of the target rows' weights.
    In each segment, covered entries get prediction + a permutation of the covered residuals, and uncovered entries a
    permutation of the uncovered values (their lookup mean is 0). Segments: one source group across all target rows
    (pool=True), or one (target row, source group) pair (pool=False; too small in the hiding rows: 2-4 weights, so a
    shuffle often leaves ARC's own pattern in place)."""
    w = arc_flat()
    base = np.zeros_like(w); covm = np.zeros(len(w), bool)
    base[covered_idx] = pred; covm[covered_idx] = True
    res = w - base
    out = w.copy()
    rng = np.random.default_rng(30_000 + seed)
    rows_idx = np.asarray(rows_idx)
    key = np.array([l.split(":")[1] for l in ROW]) if pool else ROW
    for seg in sorted(set(key[rows_idx])):
        for c in (True, False):
            ii = rows_idx[(key[rows_idx] == seg) & (covm[rows_idx] == c)]
            if len(ii) > 1:
                out[ii] = base[ii] + res[rng.permutation(ii)]
    return out
