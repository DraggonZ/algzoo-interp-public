"""Exact two-spike accuracy Z_2 of a 2nd-argmax ReLU RNN, from the weights alone.

Z_2 is the accuracy when every input except the top two is set to zero; the target stays the
position of the original 2nd largest. With max m1 at position p1 and 2nd max m2 at p2, the input
is m1 e_p1 + m2 e_p2 = r (cos θ e_p1 + sin θ e_p2) with θ in (-3π/4, π/4), so that m1 > m2.
The network has no biases, so its output depends only on θ, and it is linear on angle intervals.
We follow these intervals step by step, splitting at each ReLU's zero crossing, then intersect
with the logit-difference half-planes. The weight of an angle interval is the exact integral of
the top-two order-statistic density 90 φ(m1) φ(m2) Φ(m2)^8 over that cone (per position pair,
divide by 90). φ(m1)φ(m2) = e^{-r²/2}/2π, so the angular density depends only on sin θ:
    g(θ) = (1/2π) ∫_0^∞ r e^{-r²/2} Φ(r sin θ)^8 dr,
done by Gauss–Legendre quadrature in r and in θ. No inputs are sampled.
"""
import numpy as np
from scipy.special import ndtr

N_SEQ = 10
LO, HI = -0.75 * np.pi, 0.25 * np.pi
_r, _wr = np.polynomial.legendre.leggauss(160)
R = 6.0 * (_r + 1)            # r in [0, 12]
WR = 6.0 * _wr
_t, _wt = np.polynomial.legendre.leggauss(40)


def g_angle(theta, n_low=N_SEQ - 2):
    """Angular density of (m1, m2) for one ordered position pair."""
    s = np.sin(np.atleast_1d(theta))[:, None]
    f = R * np.exp(-R**2 / 2) * ndtr(R * s) ** n_low
    return (f * WR).sum(1) / (2 * np.pi)


def angle_mass(a, b, pieces=None):
    """∫_a^b g(θ) dθ by Gauss–Legendre on sub-pieces of width at most 0.05 rad."""
    if b <= a:
        return 0.0
    k = pieces or max(1, int(np.ceil((b - a) / 0.05)))
    edges = np.linspace(a, b, k + 1)
    tot = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        th = 0.5 * (hi - lo) * _t + 0.5 * (hi + lo)
        tot += 0.5 * (hi - lo) * (g_angle(th) * _wt).sum()
    return tot


def crossings(Bj, a, b):
    """Angles in (a, b) where Bj · (cos θ, sin θ) = 0."""
    t0 = np.arctan2(-Bj[0], Bj[1])          # Bj0 cos + Bj1 sin = 0  <=>  tan θ = -Bj0/Bj1
    out = []
    for k in range(-2, 3):
        t = t0 + k * np.pi
        if a + 1e-13 < t < b - 1e-13:
            out.append(t)
    return out


def pair_intervals(params, p1, p2, ops):
    """Linear cells for one position pair: list of (a, b, A) with h_n = A (cos θ, sin θ)."""
    w_hi, w_hh, _ = params
    d = w_hh.shape[0]
    cells = [(LO, HI, np.zeros((d, 2)))]
    for t in range(N_SEQ):
        e = np.zeros(2)
        if t == p1:
            e[0] = 1.0
        elif t == p2:
            e[1] = 1.0
        new = []
        for a, b, A in cells:
            B = w_hh @ A + np.outer(w_hi, e)
            ops[0] += d * d * 2
            cuts = sorted({c for j in range(d) for c in crossings(B[j], a, b)})
            ops[0] += d * 4
            edges = [a] + cuts + [b]
            for lo, hi in zip(edges[:-1], edges[1:]):
                mid = 0.5 * (lo + hi)
                act = B @ np.array([np.cos(mid), np.sin(mid)]) > 0
                new.append((lo, hi, B * act[:, None]))
                ops[0] += d * 2
        cells = new
    return cells


def correct_intervals(params, p1, p2, cells, ops):
    """Sub-intervals of each cell where argmax(logits) == p2 (first-index tie rule)."""
    w_oh = params[2]
    n = w_oh.shape[0]
    out = []
    for a, b, A in cells:
        L = w_oh @ A
        ops[0] += n * A.shape[0] * 2
        lo, hi = a, b
        for k in range(n):
            if k == p2:
                continue
            D = L[p2] - L[k]
            if np.abs(D).max() < 1e-12 * (1 + np.abs(L).max()):
                if k < p2:          # identical logits: argmax picks the first index
                    lo, hi = 1.0, 0.0
                    break
                continue
            # D · u(θ) > 0 on (lo, hi): find crossings, keep the positive side
            cs = crossings(D, lo, hi)
            if cs:
                c = cs[0]
                mid_left = 0.5 * (lo + c)
                if D @ np.array([np.cos(mid_left), np.sin(mid_left)]) > 0:
                    hi = c
                else:
                    lo = c
            else:
                mid = 0.5 * (lo + hi)
                if D @ np.array([np.cos(mid), np.sin(mid)]) <= 0:
                    lo, hi = 1.0, 0.0
                    break
            ops[0] += 6
        if hi > lo:
            out.append((lo, hi))
    return out


def exact_z2(params, return_detail=False):
    n = params[2].shape[0]
    assert n == N_SEQ
    ops = [0]
    total = 0.0
    detail = {}
    n_cells = 0
    for p1 in range(n):
        for p2 in range(n):
            if p1 == p2:
                continue
            cells = pair_intervals(params, p1, p2, ops)
            n_cells += len(cells)
            good = correct_intervals(params, p1, p2, cells, ops)
            acc = sum(angle_mass(lo, hi) for lo, hi in good)
            total += acc
            detail[(p1, p2)] = (acc * 90, good)
    if return_detail:
        return total, ops[0], n_cells, detail
    return total, ops[0], n_cells


if __name__ == "__main__":
    import sys
    import time

    sys.path.insert(0, "/data/output/research/algzoo/20261016_m16_10/scripts")
    from core import example

    print("normalisation (should be 1):", 90 * angle_mass(LO, HI))
    t0 = time.time()
    z, ops, nc, det = exact_z2(example(), return_detail=True)
    print(f"base Z_2 exact {z:.7f}  cells {nc}  ops {ops:.3g}  ({time.time() - t0:.1f}s)")
