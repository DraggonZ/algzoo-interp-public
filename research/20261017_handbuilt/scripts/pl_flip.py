"""Exact flip points for piecewise-linear, bias-free RNNs (h <- ReLU(W_hh h + W_hi x), scores = W_oh h).
Along x(s) = base + s * e_k, one forward pass with a tangent gives the score margin m(s) and dm/ds; inside the
current on/off pattern m is linear, so the root is s - m / (dm/ds). If a pre-activation changes sign before the
root, step just past it and repeat. Cases that do not settle fall back to bisection.
Because there are no biases, the margin at any point equals grad(margin) . x exactly (Euler), which gives each
input's exact contribution to a close call (see `contributions`)."""
import numpy as np


def forward_tangent(P, X, k):
    """Scores and their derivative with respect to x_k (per row k), plus all pre-activations and slopes."""
    wi, wh, wo = P
    n = len(X); h = np.zeros((n, len(wi))); dh = np.zeros_like(h); Z, DZ = [], []
    for t in range(X.shape[1]):
        z = h @ wh.T + X[:, t:t + 1] * wi
        dz = dh @ wh.T + (k == t)[:, None] * wi
        on = z > 0
        h, dh = np.where(on, z, 0), np.where(on, dz, 0)
        Z.append(z); DZ.append(dz)
    return h @ wo.T, dh @ wo.T, np.stack(Z, 1), np.stack(DZ, 1)


def margin(P, X, a, b):
    wi, wh, wo = P
    h = np.zeros((len(X), len(wi)))
    for t in range(X.shape[1]):
        h = np.maximum(0, h @ wh.T + X[:, t:t + 1] * wi)
    s = h @ wo.T; r = np.arange(len(X))
    return s[r, b] - s[r, a]


def bisect(P, base, k, a, b, lo, hi, steps=24):
    lo, hi = np.broadcast_to(lo, len(base)).astype(float).copy(), np.broadcast_to(hi, len(base)).astype(float).copy()
    r = np.arange(len(base))
    for _ in range(steps):
        mid = (lo + hi) / 2; X = base.copy(); X[r, k] = mid
        f = margin(P, X, a, b) > 0
        hi = np.where(f, mid, hi); lo = np.where(f, lo, mid)
    return (lo + hi) / 2


def flip(P, base, k, a, b, s0, lo, hi, max_iter=20, fallback_steps=24):
    """Value of input k at which score b overtakes score a, nearest to s0 along the search.
    Returns (flip values, passes used per row, bisection steps used per row)."""
    n = len(base); r = np.arange(n)
    lo = np.broadcast_to(lo, n).astype(float); hi = np.broadcast_to(hi, n).astype(float)
    s = np.broadcast_to(s0, n).astype(float).copy(); out = np.full(n, np.nan); done = np.zeros(n, bool)
    passes = np.zeros(n, int)
    for _ in range(max_iter):
        X = base.copy(); X[r, k] = s
        S, dS, Z, DZ = forward_tangent(P, X, k)
        m = S[r, b] - S[r, a]; dm = dS[r, b] - dS[r, a]
        step = np.where(dm != 0, -m / np.where(dm != 0, dm, 1), np.inf)
        with np.errstate(divide="ignore", invalid="ignore"):
            cross = -Z / DZ
        same = np.sign(cross) == np.sign(step)[:, None, None]
        cross = np.where(same & (np.abs(cross) > 1e-12), np.abs(cross), np.inf).reshape(n, -1).min(1)
        passes += ~done
        fin = ~done & (np.abs(step) <= cross) & (s + step >= lo) & (s + step <= hi)
        out[fin] = (s + step)[fin]; done |= fin
        if done.all():
            break
        mv = ~done
        s[mv] = np.clip(s[mv] + np.sign(step[mv]) * (cross[mv] + 1e-9), lo[mv], hi[mv])
    steps = np.zeros(n, int)
    if (~done).any():
        idx = np.where(~done)[0]
        out[idx] = bisect(P, base[idx], k[idx], a[idx], b[idx], lo[idx], hi[idx], fallback_steps)
        steps[idx] = fallback_steps
    return out, passes, steps


def contributions(P, X, a, b):
    """Exact split of the margin score_b - score_a into one term per input: c_t * x_t, with c the gradient
    inside the current on/off pattern. Sums to the margin exactly because the network has no biases."""
    wi, wh, wo = P
    n, T = X.shape; h = np.zeros((n, len(wi))); masks = []
    for t in range(T):
        z = h @ wh.T + X[:, t:t + 1] * wi; on = z > 0; h = np.where(on, z, 0); masks.append(on)
    r = np.arange(n); g = wo[b] - wo[a]                      # gradient of margin w.r.t. final h
    c = np.zeros((n, T))
    for t in range(T - 1, -1, -1):
        g = g * masks[t]                                     # through ReLU at step t
        c[:, t] = g @ wi                                     # d margin / d x_t
        g = g @ wh                                           # to h_{t-1}
    return c * X, c


def walk(P, base, k, a, b, s_start, s_end, max_iter=400):
    """Walk the line x_k = s from s_start to s_end one linear piece at a time. Inside each piece the margin
    score_b - score_a is exactly linear. Returns, per row, the list of (s_from, s_to, margin_from, margin_to)
    pieces and the number of passes (one forward pass with tangent per piece)."""
    n = len(base); r = np.arange(n); d = np.sign(s_end - s_start)
    s = np.full(n, float(s_start)); pieces = [[] for _ in range(n)]; active = np.ones(n, bool); passes = np.zeros(n, int)
    for _ in range(max_iter):
        idx = np.where(active)[0]
        if not len(idx):
            break
        X = base[idx].copy(); X[np.arange(len(idx)), k[idx]] = s[idx]
        S, dS, Z, DZ = forward_tangent(P, X, k[idx])
        ri = np.arange(len(idx))
        m = S[ri, b[idx]] - S[ri, a[idx]]; dm = dS[ri, b[idx]] - dS[ri, a[idx]]
        with np.errstate(divide="ignore", invalid="ignore"):
            cross = -Z / DZ
        cross = np.where((np.sign(cross) == d) & (np.abs(cross) > 1e-12), np.abs(cross), np.inf).reshape(len(idx), -1).min(1)
        length = np.minimum(cross, np.abs(s_end - s[idx]))
        passes[idx] += 1
        for j, i in enumerate(idx):
            pieces[i].append((s[i], s[i] + d * length[j], m[j], m[j] + d * length[j] * dm[j]))
        s[idx] = s[idx] + d * (length + 1e-9)
        active[idx] = d * (s_end - s[idx]) > 0
    return pieces, passes


def wrong_mass(pieces, F, scale=1.0, s_ref=1.0):
    """For pieces walked downward from s_ref, the probability mass of relative gaps d = (s_ref - s) / scale at which
    the margin is positive (the wrong answer wins), using the gap CDF F(d). Exact within each linear piece."""
    tot = 0.0
    for s0, s1, m0, m1 in pieces:
        lo_s, hi_s = min(s0, s1), max(s0, s1)
        if m0 > 0 and m1 > 0:
            seg = (lo_s, hi_s)
        elif m0 <= 0 and m1 <= 0:
            continue
        else:
            root = s0 + (s1 - s0) * m0 / (m0 - m1)
            seg = (root, hi_s) if (m1 > 0) == (s1 > s0) else (lo_s, root)
        tot += F((s_ref - seg[0]) / scale) - F((s_ref - seg[1]) / scale)
    return tot
