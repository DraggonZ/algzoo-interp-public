"""Tie-line quadrature, grouped design (TLQ2). Same estimate as tlq.py, but the six orderings of one set of three
positions {a, b, c} share their rest (largest value, third value, seven smaller numbers) and their tie points:
the 2-3 tie point with the largest at a (b and c both at the third's value) starts the lines of orderings (a,b,c) and
(a,c,b); the 1-2 tie point with the third at c (a and b both at the largest's value) ends the lines of (a,b,c) and
(b,a,c). So 6 tie-point forwards serve 6 lines (12 line ends) instead of 12, and the two orderings that share a tie
point have opposite leans there (one of them wins the tie), which makes their errors partly cancel.
Quadrature: 120 position sets x m points of a fixed scrambled Sobol sequence (17 dimensions as in tlq.py).
Operations: shared prefix up to the set's first position, then each tie point's forward from there, a tangent per
line end from the moving position, and forward + tangent for every further linear piece."""
import itertools, numpy as np, torch
from linelib import *
Phi = torch.special.ndtr
SETS = torch.tensor(list(itertools.combinations(range(10), 3)), device=DEV)        # 120 x 3, increasing
_cache = {}


def design(m, seed=0):
    key = (m, seed)
    if key in _cache: return _cache[key]
    eng = torch.quasirandom.SobolEngine(17, scramble=True, seed=seed)
    U = eng.draw(120 * m).to(DEV, dt).clamp(1e-12, 1 - 1e-12)
    V = torch.special.ndtri(U[:, :10]).sort(1, descending=True).values
    perm = U[:, 10:].argsort(1)
    S3 = SETS.repeat_interleave(m, 0); n = len(S3); r = torch.arange(n, device=DEV)
    free = torch.ones(n, 10, dtype=torch.bool, device=DEV)
    for j in range(3): free[r, S3[:, j]] = False
    fpos = torch.nonzero(free)[:, 1].reshape(n, 7)
    Z = torch.zeros(n, 10, device=DEV, dtype=dt); Z.scatter_(1, fpos, V[:, 3:].gather(1, perm))
    _cache[key] = (Z, S3, V[:, 0], V[:, 2]); return _cache[key]


def estimate(W, m=1, seed=0, max_pieces=40, return_parts=False):
    Z, S3, L, T = design(m, seed); n = len(Z); r = torch.arange(n, device=DEV)
    first = S3[:, 0].to(dt)
    flops = (first * F_STEP).sum().item()                     # shared prefix, once per group
    tot = Phi(L) - Phi(T)
    q = torch.zeros(n, 6, device=DEV, dtype=dt)               # per ordering (pL, pS, pT) in itertools.permutations order of (0,1,2)
    orders = list(itertools.permutations(range(3)))
    oi = {o: i for i, o in enumerate(orders)}
    never = torch.zeros(n, 6, dtype=torch.bool, device=DEV)
    # 2-3 tie points: largest at slot i, the other two at T; lines move one of the two up
    for i in range(3):
        j, k = [s for s in range(3) if s != i]
        X0 = Z.clone(); X0[r, S3[:, i]] = L; X0[r, S3[:, j]] = T; X0[r, S3[:, k]] = T
        flops += ((10 - first) * F_STEP).sum().item() + n * F_OUT
        for mv, other in ((j, k), (k, j)):
            pS = S3[:, mv]
            w, _, f = first_win(W, X0, pS, pS, L - T, +1.0, max_pieces); flops += f
            o = oi[(i, mv, other)]; q[:, o] += (Phi(T + w) - Phi(T)) / tot; never[:, o] = w >= (L - T) * (1 - 1e-12)
    # 1-2 tie points: third at slot k, the other two at L; lines move one of the two down
    for k in range(3):
        i, j = [s for s in range(3) if s != k]
        X1 = Z.clone(); X1[r, S3[:, k]] = T; X1[r, S3[:, i]] = L; X1[r, S3[:, j]] = L
        flops += ((10 - first) * F_STEP).sum().item() + n * F_OUT
        for lg, mv in ((i, j), (j, i)):
            pS = S3[:, mv]
            w2, _, f = first_win(W, X1, pS, pS, L - T, -1.0, max_pieces); flops += f
            o = oi[(lg, mv, k)]; q[:, o] += torch.where(never[:, o], torch.zeros_like(w2), (Phi(L) - Phi(L - w2)) / tot)
    q = q.clamp(0, 1); acc = 1 - q.mean().item()
    if return_parts: return acc, flops, dict(q=q, never=never)
    return acc, flops
