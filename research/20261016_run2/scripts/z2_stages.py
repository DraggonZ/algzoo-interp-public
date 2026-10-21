"""Exact decomposition of two-spike errors into write and carry failures, from the weights.

For each position pair, T = max(p1, p2) is the step of the last spike. The running pointer at step t is
argmax(W_oh h_t) in padded slots (input s sits at slot s + 9 - t after t+1 inputs). For every angle interval
we record whether the pointer is right at every step from T to the end. Masses are exact (angle quadrature).
Output per model: mass right at T and at the end, lost after T (carry), recovered after T, and the step of loss.
"""
import sys

import numpy as np

from algzoo.exact_z2 import HI, LO, N_SEQ, angle_mass, crossings


def follow(params, p1, p2):
    """Cells (a, b, maps) with maps[t] = A_t for t >= T, h_t = A_t u(θ)."""
    w_hi, w_hh, _ = params
    d = w_hh.shape[0]
    T = max(p1, p2)
    cells = [(LO, HI, np.zeros((d, 2)), {})]
    for t in range(N_SEQ):
        e = np.zeros(2)
        if t == p1:
            e[0] = 1.0
        elif t == p2:
            e[1] = 1.0
        new = []
        for a, b, A, maps in cells:
            B = w_hh @ A + np.outer(w_hi, e)
            cuts = sorted({c for j in range(d) for c in crossings(B[j], a, b)})
            edges = [a] + cuts + [b]
            for lo, hi in zip(edges[:-1], edges[1:]):
                mid = 0.5 * (lo + hi)
                act = B @ np.array([np.cos(mid), np.sin(mid)]) > 0
                A2 = B * act[:, None]
                m2 = dict(maps)
                if t >= T:
                    m2[t] = A2
                new.append((lo, hi, A2, m2))
        cells = new
    return cells, T


def right_set(L, target, lo, hi):
    """Sub-interval of (lo, hi) where argmax(L u) == target (first-index ties)."""
    for k in range(L.shape[0]):
        if k == target:
            continue
        D = L[target] - L[k]
        if np.abs(D).max() < 1e-12 * (1 + np.abs(L).max()):
            if k < target:
                return None
            continue
        cs = crossings(D, lo, hi)
        if cs:
            c = cs[0]
            ml = 0.5 * (lo + c)
            if D @ np.array([np.cos(ml), np.sin(ml)]) > 0:
                hi = c
            else:
                lo = c
        else:
            mid = 0.5 * (lo + hi)
            if D @ np.array([np.cos(mid), np.sin(mid)]) <= 0:
                return None
    return (lo, hi) if hi > lo else None


def stages(params):
    w_oh = params[2]
    out = {"right_T_right_end": 0.0, "right_T_wrong_end": 0.0, "wrong_T_right_end": 0.0, "wrong_T_wrong_end": 0.0}
    loss_step = np.zeros(N_SEQ)          # mass first lost at step t (right at T, wrong at t)
    per_pair = {}
    for p1 in range(N_SEQ):
        for p2 in range(N_SEQ):
            if p1 == p2:
                continue
            cells, T = follow(params, p1, p2)
            pp = np.zeros(4)
            for a, b, A, maps in cells:
                # breakpoints of correctness at every step t >= T inside (a, b)
                pts = {a, b}
                sets = {}
                for t in range(T, N_SEQ):
                    s = right_set(w_oh @ maps[t], p2 + N_SEQ - 1 - t, a, b)
                    sets[t] = s
                    if s:
                        pts.update(s)
                pts = sorted(pts)
                for lo, hi in zip(pts[:-1], pts[1:]):
                    mid = 0.5 * (lo + hi)
                    ok = [sets[t] is not None and sets[t][0] <= mid <= sets[t][1] for t in range(T, N_SEQ)]
                    m = angle_mass(lo, hi)
                    key = ("right_T_" if ok[0] else "wrong_T_") + ("right_end" if ok[-1] else "wrong_end")
                    out[key] += m
                    pp[["right_T_right_end", "right_T_wrong_end", "wrong_T_right_end", "wrong_T_wrong_end"].index(key)] += m
                    if ok[0] and not all(ok):
                        loss_step[T + ok.index(False)] += m
            per_pair[(p1, p2)] = pp * 90
    return out, loss_step, per_pair


if __name__ == "__main__":
    from algzoo.bench import load_bench
    from algzoo.core import example

    names, acc, models = load_bench(("mult",))
    sel = sys.argv[1].split(",") if len(sys.argv) > 1 else ["example"] + list(names)
    res = {}
    for nm in sel:
        params = example() if nm == "example" else models[list(names).index(nm)]
        out, ls, pp = stages(params)
        res[nm] = (out, ls)
        print(f"{nm:14s} Z2 {out['right_T_right_end'] + out['wrong_T_right_end']:.4f} | right at write {out['right_T_right_end'] + out['right_T_wrong_end']:.4f}"
              f" | lost after write {out['right_T_wrong_end']:.4f} | recovered {out['wrong_T_right_end']:.4f}"
              f" | first loss by step {np.round(ls, 3)}", flush=True)
    if len(sel) > 5:
        np.save("/data/output/research/algzoo/20261016_run2/reports/z2_stages.npy", res, allow_pickle=True)
