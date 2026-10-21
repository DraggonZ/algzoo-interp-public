"""Shared pieces for the cheap sampling-free estimates (cheap/). Everything here is the mean-background reduction of
propagate/: the three largest numbers exact (values m1 > m2 > m3 at positions p1, p2, p3), the other seven at the
mean of a standard normal truncated below m3. A 'run' is one forward pass of the network on one 10-number input."""
import itertools
import numpy as np, torch
from scipy.stats import norm
from algzoo.core import load, example

DEV = "cuda"; dt = torch.float64
T = torch.tensor(list(itertools.permutations(range(10), 3)), device=DEV)       # [720, 3]: p1, p2, p3
SEEDS_DEV = list(range(200, 216)); SEEDS_HELD = list(range(100, 116))
TRUE = np.load("/data/output/research/algzoo/20261017_handbuilt/hilton/propagate/across.npy")[:, 0]  # ARC, 100-115, 200-215
REF = np.load("/data/output/research/algzoo/20261017_handbuilt/hilton/propagate/across.npy")[:, 1]   # converged reduction


def nets():
    """[(name, W)] in the order of across.npy: ARC, 100-115, 200-215."""
    out = [("ARC", example())] + [(str(s), load(f"/data/output/research/algzoo/20261016_run3/models/2nd_argmax_16_10_1073741824_{s}.pth"))
                                  for s in SEEDS_HELD + SEEDS_DEV]
    return [(n, tuple(torch.tensor(np.asarray(q, float), device=DEV, dtype=dt) for q in p)) for n, p in out]


def bg(m3):
    """Mean of a standard normal truncated below m3 (torch)."""
    m3 = torch.as_tensor(m3, device=DEV, dtype=dt)
    pdf = torch.exp(-m3 * m3 / 2) / np.sqrt(2 * np.pi); cdf = 0.5 * (1 + torch.erf(m3 / np.sqrt(2)))
    return -pdf / cdf.clamp_min(1e-300)


def inputs(pos, val):
    """pos [n,3] long, val [n,3] (m1, m2, m3) -> X [n,10]."""
    n = pos.shape[0]; ar = torch.arange(n, device=DEV)
    X = bg(val[:, 2])[:, None].repeat(1, 10)
    for k in range(3): X[ar, pos[:, k]] = val[:, k]
    return X


def scores(W, X, dX=None):
    """Scores [n,10]; with dX also their derivative along dX (forward tangent, ReLU pattern of X)."""
    wi, wh, wo = W; n = X.shape[0]
    h = torch.zeros(n, wi.shape[0], device=DEV, dtype=dt); dh = torch.zeros_like(h) if dX is not None else None
    for t in range(10):
        z = h @ wh.T + X[:, t:t + 1] * wi; on = z > 0
        if dX is not None: dh = torch.where(on, dh @ wh.T + dX[:, t:t + 1] * wi, 0.0)
        h = torch.where(on, z, 0.0)
    return (h @ wo.T, dh @ wo.T if dX is not None else None)


def margin_from(S, p2, dS=None):
    """Second-largest's score minus the best other; with dS also its derivative."""
    n = S.shape[0]; ar = torch.arange(n, device=DEV)
    O = S.clone(); O[ar, p2] = -1e300; k = O.argmax(1)
    m = S[ar, p2] - S[ar, k]
    return (m, None if dS is None else dS[ar, p2] - dS[ar, k], k)


def npdf(x): return np.exp(-x * x / 2) / np.sqrt(2 * np.pi)
def ncdf(x): return norm.cdf(x)


def density(m1, m2, m3):
    """Joint density of the ordered top three of 10 i.i.d. normals, per ordered position triple (sums to 1 over
    the 720 triples): 7!/... -> for one triple: phi(m1) phi(m2) phi(m3) Phi(m3)^7 (the 720 triples' integrals add to 1)."""
    return npdf(m1) * npdf(m2) * npdf(m3) * ncdf(m3) ** 7


class Log:
    def __init__(self, path): self.f = open(path, "w")
    def __call__(self, *a):
        s = " ".join(str(x) for x in a); print(s, flush=True); self.f.write(s + "\n"); self.f.flush()
