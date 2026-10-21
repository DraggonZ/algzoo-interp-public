"""Exact line integrals for the bias-free ReLU RNN (h <- relu(W_hh h + w_in x), scores = W_oh h).

Along x(a) = base + a * e_k (one input moving), the network is piecewise linear in a. One forward pass with a tangent
gives every pre-activation z and its slope dz, so the end of the current linear piece is the first a where some
pre-activation changes sign; within the piece every score is linear in a, so where a given position has the highest
score is an exact interval. Walking piece by piece gives the exact set of a where the network names position c.

Cost accounting (floating-point operations): a forward step costs 2*16*16 + 2*16 + 16 (matvec, input, relu) = 560;
the tangent shares the switch pattern and costs 2*16*16 + 16 = 528 from the step where the moving input enters; the
readout costs 2*10*16 = 320 (tangent 320 more). Sampling a list costs 10*560 + 320 = 5,920 operations."""
import numpy as np, torch
DEV = "cuda"; dt = torch.float64
F_STEP, F_TAN, F_OUT = 560, 528, 320
F_LIST = 10 * F_STEP + F_OUT


def fwd_tan(W, X, k):
    """X [n,10], k [n] moving position. Returns scores, d scores, the a-step to the next switch (inf if none)
    per row, given a = 0 at X and moving in +a direction (dir_sign = +1)."""
    wi, wh, wo = W
    n = len(X); h = torch.zeros(n, 16, device=DEV, dtype=dt); dh = torch.zeros_like(h)
    nxt = torch.full((n,), float("inf"), device=DEV, dtype=dt)
    ar = torch.arange(10, device=DEV)
    for t in range(10):
        z = h @ wh.T + X[:, t:t + 1] * wi
        dz = dh @ wh.T + (k == t).to(dt)[:, None] * wi
        on = z > 0
        # distance to sign change in +a direction: z + dz*a = 0, a = -z/dz > 0
        r = -z / dz
        r = torch.where((r > 1e-12) & torch.isfinite(r), r, torch.full_like(r, float("inf")))
        nxt = torch.minimum(nxt, r.min(1).values)
        h = torch.where(on, z, torch.zeros_like(z)); dh = torch.where(on, dz, torch.zeros_like(dz))
    return h @ wo.T, dh @ wo.T, nxt


def win_interval(s, ds, c, lo, hi):
    """Within one piece (a in [lo, hi], scores s + ds*(a - lo)), the interval where position c is highest.
    Returns (l, u) with l >= u meaning empty."""
    n = len(s); r = torch.arange(n, device=DEV)
    m = s[r, c][:, None] - s; dm = ds[r, c][:, None] - ds          # margin of c over j at a = lo: m + dm (a - lo) > 0
    m[r, c] = 1.0; dm[r, c] = 0.0
    # each j: if dm > 0: a - lo > -m/dm ; if dm < 0: a - lo < -m/dm ; if dm == 0: need m > 0
    root = -m / torch.where(dm == 0, torch.ones_like(dm), dm)
    lower = torch.where(dm > 0, root, torch.where((dm == 0) & (m <= 0), torch.full_like(m, float("inf")), torch.full_like(m, -float("inf"))))
    upper = torch.where(dm < 0, root, torch.full_like(m, float("inf")))
    l = torch.maximum(lower.max(1).values + lo, lo); u = torch.minimum(upper.min(1).values + lo, hi)
    return l, u


def walk(W, X, k, c, amax, max_pieces=200):
    """Along X + a e_k for a in [0, amax] (amax per row), list of (row, lo, hi) stretches where position c wins.
    Returns the stretches as tensors, and the number of pieces and operation count used."""
    n = len(X); a = torch.zeros(n, device=DEV, dtype=dt); act = torch.arange(n, device=DEV)
    rows, los, his = [], [], []; flops = 0; pieces = 0
    ek = torch.nn.functional.one_hot(k, 10).to(dt)
    for it in range(max_pieces):
        if len(act) == 0: break
        Xa = X[act] + a[act, None] * ek[act]
        s, ds, nxt = fwd_tan(W, Xa, k[act])
        # operations: forward all 10 steps + tangent from the moving input's step
        flops += int(len(act) * (10 * F_STEP + 2 * F_OUT) + ((10 - k[act]).to(dt) * F_TAN).sum().item())
        pieces += len(act)
        lo = a[act]; hi = torch.minimum(lo + nxt * (1 + 1e-12) + 1e-13, amax[act])
        l, u = win_interval(s, ds, c[act], lo, hi)
        ok = u > l
        rows.append(act[ok]); los.append(l[ok]); his.append(u[ok])
        a[act] = hi
        act = act[hi < amax[act]]
    return (torch.cat(rows), torch.cat(los), torch.cat(his)), pieces, flops


def first_win(W, X, k, c, amax, sign=1.0, max_pieces=12):
    """Along X + sign * a * e_k, a in [0, amax]: the first a where position c is highest (amax if never within
    max_pieces pieces; the cap is reported). Returns a_first, pieces used per row, operation count."""
    n = len(X); a = torch.zeros(n, device=DEV, dtype=dt); done = torch.zeros(n, dtype=torch.bool, device=DEV)
    res = amax.clone(); flops = 0; npc = torch.zeros(n, device=DEV, dtype=dt); act = torch.arange(n, device=DEV)
    ek = torch.nn.functional.one_hot(k, 10).to(dt) * sign
    for it in range(max_pieces):
        if len(act) == 0: break
        Xa = X[act] + a[act, None] * ek[act]
        s, ds, nxt = fwd_tan_dir(W, Xa, k[act], sign)
        # cost: the first piece shares the forward pass at the tie (counted by the caller); later pieces recompute
        # from the moved position on (forward + tangent); here we count tangent from k on, and forward from k on
        # for pieces after the first
        kk = k[act].to(dt)
        flops += int(((10 - kk) * F_TAN).sum().item() + len(act) * F_OUT)
        if it > 0: flops += int(((10 - kk) * F_STEP).sum().item() + len(act) * F_OUT)
        npc[act] += 1
        lo = a[act]; hi = torch.minimum(lo + nxt * (1 + 1e-12) + 1e-13, amax[act])
        l, u = win_interval(s, ds, c[act], lo, hi)
        ok = u > l
        res[act[ok]] = l[ok]
        a[act] = hi
        act = act[(~ok) & (hi < amax[act])]
    return res, npc, flops


def fwd_tan_dir(W, X, k, sign):
    wi, wh, wo = W
    n = len(X); h = torch.zeros(n, 16, device=DEV, dtype=dt); dh = torch.zeros_like(h)
    nxt = torch.full((n,), float("inf"), device=DEV, dtype=dt)
    for t in range(10):
        z = h @ wh.T + X[:, t:t + 1] * wi
        dz = dh @ wh.T + (k == t).to(dt)[:, None] * wi * sign
        on = z > 0
        r = -z / dz
        r = torch.where((r > 1e-12) & torch.isfinite(r), r, torch.full_like(r, float("inf")))
        nxt = torch.minimum(nxt, r.min(1).values)
        h = torch.where(on, z, torch.zeros_like(z)); dh = torch.where(on, dz, torch.zeros_like(dz))
    return h @ wo.T, dh @ wo.T, nxt
