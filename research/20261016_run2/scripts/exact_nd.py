"""Exact accuracy of a 2nd-argmax ReLU RNN for any sequence length n, without sampling inputs.

The network has no biases, so it is linear on polyhedral cones in R^n and the Gaussian measure of a cone
is its solid-angle fraction. Cells are pointed polyhedral cones kept in both forms: extreme rays R and
constraint normals H (H x >= 0), updated by the double-description rule when a hyperplane cuts a cell.
Steps: start from the 2^n orthants; at each input step cut by every neuron's pre-activation sign; at the
end cut by the order planes x_i = x_j and by the logit-difference planes of the target; triangulate each
correct cone into simplicial cones and integrate. The measure of a simplicial cone with unit generators V is
    |det V| / area(S^{n-1}) * ∫_simplex |V μ|^{-n} dμ,
computed by Gauss–Legendre on the Duffy-collapsed simplex after splitting the cone until it is narrow.
Logit ties follow argmax's first-index rule. Multiply-adds are counted roughly.
"""
import itertools
import math

import numpy as np
from scipy.spatial import Delaunay

from gm_rule import gm_rule

TOL = 1e-10
ROBUST = False           # add the algebraic adjacency test (see cut); too costly, off
SNAP = 0                 # (off: breaks degenerate cells) rays this close to a cutting plane are put on it; near-duplicate new rays are merged


class Ops:
    def __init__(self):
        self.n = 0


def cut(R, H, b, ops):
    """Split cone (R, H) by the hyperplane b·x = 0. Returns (plus, minus); each is (R, H) or None."""
    k, n = R.shape
    b = b / np.linalg.norm(b)
    s = R @ b
    ops.n += k * n
    if SNAP:
        near = (np.abs(s) <= SNAP) & (np.abs(s) > TOL)
        if near.any():                                # project nearly-on-plane rays onto the plane
            R = R.copy()
            R[near] -= s[near, None] * b
            R[near] /= np.linalg.norm(R[near], axis=1, keepdims=True)
            s = s.copy()
            s[near] = 0.0
    pos, neg = s > TOL, s < -TOL
    if not neg.any():
        return (R, H), None
    if not pos.any():
        return None, (R, H)
    D = np.abs(R @ H.T)
    T = D < 1e-11                                     # tight constraints per ray
    ops.n += k * H.shape[0] * n
    ip, jn = np.flatnonzero(pos), np.flatnonzero(neg)
    ii, jj = np.repeat(ip, len(jn)), np.tile(jn, len(ip))
    common = T[ii] & T[jj]                            # [P, m]
    ops.n += len(ii) * H.shape[0]
    adj = np.zeros(len(ii), dtype=bool)
    ok = common.sum(1) >= n - 2
    if ok.any():
        # combinatorial test: adjacent iff no third ray is tight on every constraint common to rays i and j
        miss = (~T).astype(np.int32) @ common[ok].T.astype(np.int32)      # [k, P]; 0 = contains
        ops.n += k * H.shape[0] * int(ok.sum())      # counted since review 3 (earlier logs omit it: about +37%)
        adj[ok] = (miss == 0).sum(0) == 2
    if ROBUST and (~adj).any():
        # algebraic test with a looser tightness: constraints touching both rays have rank n-2. A false accept only
        # adds a point inside the cell (volume unchanged); a missed adjacency would lose volume.
        Tl = D < 1e-8
        rest = np.flatnonzero(~adj)
        cl = (Tl[ii[rest]] & Tl[jj[rest]]).astype(float)                  # [P', m]
        cand = cl.sum(1) >= n - 2
        if cand.any():
            r2 = rest[cand]
            G = np.einsum("pc,ci,cj->pij", cl[cand], H, H)
            ev = np.linalg.eigvalsh(G)                                     # ascending
            adj[r2] = ev[:, -(n - 2)] > 1e-12 * np.maximum(ev[:, -1], 1e-300)
            ops.n += len(r2) * (H.shape[0] * n * n + n ** 3)
    ii, jj = ii[adj], jj[adj]
    new = []
    if len(ii):
        if len(ii):
            new = s[ii, None] * R[jj] - s[jj, None] * R[ii]
            new /= np.linalg.norm(new, axis=1, keepdims=True)
            if SNAP and len(new) > 1:                 # merge near-duplicate new rays
                keep = np.ones(len(new), dtype=bool)
                G = new @ new.T
                for a_ in range(len(new)):
                    if keep[a_]:
                        dup = (G[a_] > 1 - 1e-14) & keep
                        dup[a_] = False
                        keep[dup] = False
                new = new[keep]
            ops.n += len(ii) * 3 * n
    zer = ~pos & ~neg
    Rp = np.vstack([R[pos | zer]] + ([new] if len(new) else []))
    Rm = np.vstack([R[neg | zer]] + ([new] if len(new) else []))
    plus = (Rp, np.vstack([H, b])) if len(Rp) >= n else None
    minus = (Rm, np.vstack([H, -b])) if len(Rm) >= n else None
    return plus, minus


def orthants(n):
    cells = []
    for signs in itertools.product((1.0, -1.0), repeat=n):
        D = np.diag(signs)
        cells.append((D.copy(), D.copy()))          # rays = constraint normals = signed unit vectors
    return cells


# ---------------------------------------------------------------- measure of cones
def duffy_rule(dim, q):
    """Exact Duffy map for the simplex {μ >= 0, Σμ <= 1} in R^dim: μ_i = u_i Π_{j<i} (1-u_j),
    Jacobian Π_{i=1}^{dim-1} (1-u_i)^{dim-i}."""
    x, w = np.polynomial.legendre.leggauss(q)
    x, w = (x + 1) / 2, w / 2
    nodes, weights = [], []
    for idx in itertools.product(range(q), repeat=dim):
        u = x[list(idx)]
        wt = np.prod(w[list(idx)])
        mu = np.zeros(dim)
        rem = 1.0
        for i in range(dim):
            mu[i] = rem * u[i]
            rem *= 1 - u[i]
        for i in range(dim - 1):
            wt *= (1 - u[i]) ** (dim - 1 - i)
        nodes.append(mu)
        weights.append(wt)
    return np.array(nodes), np.array(weights)


_RULES = {}


MAX_ANGLE, Q_PTS = 0.5, 3
RULE = "gm"            # "gm": Grundmann–Möller, degree 3 for pieces narrower than NARROW rad, else degree 7
GM_S = None            # fixed GM degree 2*GM_S+1 overrides the adaptive choice
NARROW = 0.2
NARROW_S = 0             # GM degree 2*NARROW_S+1 on narrow pieces (0 = centroid rule)


def simplicial_measure(V, ops, max_angle=None, q=None):
    """Gaussian measure of the cone spanned by the unit columns of V (n x n)."""
    max_angle = MAX_ANGLE if max_angle is None else max_angle
    q = Q_PTS if q is None else q
    n = V.shape[0]
    G = V.T @ V
    iu = np.triu_indices(n, 1)
    cosmin = G[iu].min()
    if cosmin < math.cos(max_angle):
        i, j = iu[0][np.argmin(G[iu])], iu[1][np.argmin(G[iu])]
        m = V[:, i] + V[:, j]
        m /= np.linalg.norm(m)
        V1, V2 = V.copy(), V.copy()
        V1[:, i] = m
        V2[:, j] = m
        ops.n += n * n
        return simplicial_measure(V1, ops, max_angle, q) + simplicial_measure(V2, ops, max_angle, q)
    gs = GM_S if GM_S is not None else (NARROW_S if cosmin > math.cos(NARROW) else 3)
    key = (n, q, RULE, gs)
    if key not in _RULES:
        if RULE == "gm":
            full, w = gm_rule(n - 1, gs)
        else:
            nodes, w = duffy_rule(n - 1, q)
            full = np.c_[nodes, 1 - nodes.sum(1)]
        _RULES[key] = (full, w)
    full, w = _RULES[key]
    pts = full @ V.T                                  # [Q, n]
    val = (w * np.linalg.norm(pts, axis=1) ** (-n)).sum()
    ops.n += len(w) * n * (n + 1)
    area = 2 * math.pi ** (n / 2) / math.gamma(n / 2)
    return abs(np.linalg.det(V)) * val / area


def cone_measure(R, ops):
    """Measure of a pointed polyhedral cone with unit extreme rays R (k x n)."""
    k, n = R.shape
    if k == n:
        return simplicial_measure(R.T, ops)
    a = R.mean(0)
    a /= np.linalg.norm(a)
    P = R / (R @ a)[:, None]
    Q = np.linalg.svd(np.eye(n) - np.outer(a, a))[0][:, : n - 1]
    Y = P @ Q
    ops.n += k * n * n
    try:
        simps = Delaunay(Y).simplices
        if simps.max() >= k:                          # qhull's point at infinity on cospherical input
            raise ValueError
    except Exception:
        try:
            simps = Delaunay(Y, qhull_options="QJ").simplices   # joggled input: relative volume error ~1e-11
        except Exception:
            return 0.0                                # degenerate (lower-dimensional) cone
    tot = 0.0
    for simp in simps:
        tot += simplicial_measure(R[simp].T, ops)
    return tot


# ---------------------------------------------------------------- the network
def mass_proxy(R, ops):
    """Cheap size proxy for a cone with unit rays R: sqrt(det(R^T R / k)), ~ angular radius^(n-1)."""
    k, n = R.shape
    ops.n += k * n * n + n ** 3
    return math.sqrt(max(np.linalg.det(R.T @ R / k), 0.0))


def exact_accuracy(params, return_total=False, rank=2, prune=0.0):
    """Target: position of the rank-th largest input (2 = 2nd argmax; (n+1)//2 = argmedian for odd n).
    Depth-first: each cell is followed through all remaining steps before the next, so memory stays small."""
    w_hi, w_hh, w_oh = params
    n = w_oh.shape[0]
    d = w_hh.shape[0]
    ops = Ops()
    order_planes = [(i, j) for i in range(n) for j in range(i + 1, n)]
    acc = {"regions": 0, "wrong": 0.0, "wrong_pieces": 0, "correct": 0.0, "total": 0.0, "ops_measure": 0, "pruned": 0,
           "skipped_proxy": 0.0, "small_proxy": 0.0, "small_wrong": 0.0}

    def finish(R, H, A):
        acc["regions"] += 1
        wrong_before = acc["wrong"]
        L = w_oh @ A
        ops.n += n * d * n
        pieces = [(R, H)]
        for i, j in order_planes:
            v = np.zeros(n)
            v[i], v[j] = 1.0, -1.0
            nxt = []
            for R2, H2 in pieces:
                pl, mi = cut(R2, H2, v, ops)
                nxt += [c for c in (pl, mi) if c is not None]
            pieces = nxt
        for R2, H2 in pieces:
            tgt = int(np.argsort(R2.mean(0))[-rank])  # position of the rank-th largest on this order cell
            cur = (R2, H2)
            wrong_here = []
            for k in range(n):
                if k == tgt or cur is None:
                    continue
                D = L[tgt] - L[k]
                if np.abs(D).max() < 1e-12 * (1 + np.abs(L).max()):
                    if k < tgt:                        # identical logits: argmax picks the first index
                        wrong_here.append(cur)
                        cur = None
                    continue
                cur, bad = cut(cur[0], cur[1], D, ops)
                if bad is not None:                   # disjoint convex parts of the wrong set
                    wrong_here.append(bad)
            before = ops.n
            for P in wrong_here:
                acc["wrong"] += cone_measure(P[0], ops)
            acc["wrong_pieces"] += len(wrong_here)
            if return_total:
                acc["total"] += cone_measure(R2, ops)
                if cur is not None:
                    acc["correct"] += cone_measure(cur[0], ops)
            acc["ops_measure"] += ops.n - before
        if prune:                                          # wrong mass per unit size proxy among the smallest explored regions
            pf = mass_proxy(R, ops)
            if pf < 10 * prune:
                acc["small_proxy"] += pf
                acc["small_wrong"] += acc["wrong"] - wrong_before

    def step(R, H, A, t):
        if t == n:
            finish(R, H, A)
            return
        B = w_hh @ A
        B[:, t] += w_hi
        ops.n += d * d * n
        pieces = [(R, H, np.ones(d, dtype=bool))]
        for j in range(d):
            nxt = []
            for R2, H2, act in pieces:
                if np.abs(B[j]).max() < 1e-14:
                    a2 = act.copy()
                    a2[j] = False
                    nxt.append((R2, H2, a2))
                    continue
                pl, mi = cut(R2, H2, B[j], ops)
                if pl is not None:
                    nxt.append((pl[0], pl[1], act))
                if mi is not None:
                    a2 = act.copy()
                    a2[j] = False
                    nxt.append((mi[0], mi[1], a2))
            pieces = nxt
        for R2, H2, act in pieces:
            if prune:
                pc = mass_proxy(R2, ops)
                if pc < prune:                             # budgeted mode: skip small cells
                    acc["pruned"] += 1
                    acc["skipped_proxy"] += pc
                    continue
            step(R2, H2, B * act[:, None], t + 1)

    for R, H in orthants(n):
        step(R, H, np.zeros((d, n)), 0)
    stats = {"regions": acc["regions"], "wrong_pieces": acc["wrong_pieces"], "ops": ops.n, "pruned": acc["pruned"],
             "ops_enum": ops.n - acc["ops_measure"]}
    if return_total:
        stats["correct_direct"], stats["total"] = acc["correct"], acc["total"]
    if prune:
        rate = acc["small_wrong"] / acc["small_proxy"] if acc["small_proxy"] > 0 else 0.0
        stats["corrected"] = 1.0 - acc["wrong"] - rate * acc["skipped_proxy"]
    return 1.0 - acc["wrong"], stats


if __name__ == "__main__":
    import sys
    import time

    from algzoo.core import load
    from algzoo.exact_s2 import exact_accuracy as exact3

    for name in [a for a in sys.argv[1:] if not a.startswith("--")] or ["2nd_argmax_2_3_1073741824_bo5.pth", "2nd_argmax_4_3_1073741824_bo5.pth"]:
        params = load(name)
        t0 = time.time()
        rank = (params[2].shape[0] + 1) // 2 if "argmedian" in name else 2
        acc, st = exact_accuracy(params, return_total="--check" in sys.argv, rank=rank)
        msg = (f"{name}: exact {acc:.8f} regions {st['regions']} wrong pieces {st['wrong_pieces']}"
               f" ops {st['ops']:.3g} (enumeration {st['ops_enum']:.3g}) ({time.time() - t0:.1f}s)")
        if "total" in st:
            msg += f" | direct {st['correct_direct']:.8f}, total mass {st['total']:.8f}"
        if params[2].shape[0] == 3:
            msg += f"  | exact_s2 {exact3(*params)[0]:.8f}"
        print(msg, flush=True)
