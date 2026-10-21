"""Tie-line quadrature (TLQ): a deterministic accuracy estimate from the weights.

Accuracy = 1 - E[wrong]. For a list, move the second-largest along its whole range (from the third up to the largest)
with everything else fixed. The network is piecewise linear along this line, so the stretches where it names the
wrong position are found exactly, piece by piece, and their probability under the normal density is closed form.
Almost all of the wrong mass (97-99%) lies in the stretch touching either end of the line (next to the 2-3 tie or
the 1-2 tie), so only those two stretches are found (`first_win` from each end).
Quadrature over everything else: all 720 arrangements of the three largest's positions (equal weight); within each,
m points for the values (largest, third, seven smaller) from a fixed scrambled Sobol sequence in 17 dimensions:
10 normals sorted give the order statistics, 7 more coordinates fix which smaller value goes to which free position.
Operations are counted as in linelib (forward shared up to the moving position, both ends, tangents, extra pieces)."""
import numpy as np, torch
from linelib import *
Phi = torch.special.ndtr
_cache = {}


def design(m, seed=0):
    key = (m, seed)
    if key in _cache: return _cache[key]
    eng = torch.quasirandom.SobolEngine(17, scramble=True, seed=seed)
    U = eng.draw(720 * m).to(DEV, dt).clamp(1e-12, 1 - 1e-12)
    V = torch.special.ndtri(U[:, :10]).sort(1, descending=True).values      # order statistics v1 > ... > v10
    perm = U[:, 10:].argsort(1)                                               # which smaller value goes where
    import itertools
    arr = torch.tensor(list(itertools.permutations(range(10), 3)), device=DEV)   # (pL, pS, pT)
    A = arr.repeat_interleave(m, 0)
    n = len(A); X = torch.zeros(n, 10, device=DEV, dtype=dt); r = torch.arange(n, device=DEV)
    free = torch.ones(n, 10, dtype=torch.bool, device=DEV); free[r, A[:, 0]] = False; free[r, A[:, 1]] = False; free[r, A[:, 2]] = False
    fpos = torch.nonzero(free)[:, 1].reshape(n, 7)
    X.scatter_(1, fpos, V[:, 3:].gather(1, perm))
    X[r, A[:, 0]] = V[:, 0]; X[r, A[:, 2]] = V[:, 2]
    _cache[key] = (X, A, V[:, 0], V[:, 2]); return _cache[key]


def estimate(W, m=1, seed=0, max_pieces=40, both_ends=True, return_parts=False):
    X, A, L, T = design(m, seed); n = len(X); r = torch.arange(n, device=DEV); pS = A[:, 1]
    X0 = X.clone(); X0[r, pS] = T
    w, n1, f1 = first_win(W, X0, pS, pS, L - T, +1.0, max_pieces)
    never = w >= (L - T) * (1 - 1e-12)
    tot = Phi(L) - Phi(T); q23 = (Phi(T + w) - Phi(T)) / tot
    flops = (pS.to(dt) * F_STEP).sum().item() + f1 + n * ((10 - pS.to(dt)).mean().item() * F_STEP + F_OUT)
    q12 = torch.zeros_like(q23)
    if both_ends:
        X1 = X.clone(); X1[r, pS] = L
        w2, n2, f2 = first_win(W, X1, pS, pS, L - T, -1.0, max_pieces)
        q12 = torch.where(never, torch.zeros_like(w2), (Phi(L) - Phi(L - w2)) / tot)
        flops += f2 + ((10 - pS.to(dt)) * F_STEP).sum().item() + n * F_OUT
    q = (q23 + q12).clamp(0, 1)
    acc = 1 - q.mean().item()
    if return_parts: return acc, flops, dict(q23=q23, q12=q12, never=never, X=X, A=A, L=L, T=T)
    return acc, flops
