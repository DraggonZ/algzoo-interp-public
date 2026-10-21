"""Shared pieces for the small-numbers-as-noise estimate.

Setting: the network reads ten iid N(0,1) numbers. Given the three largest (values m1 > m2 > m3, positions p1, p2, p3),
the other seven are iid N(0,1) truncated below m3. The decision quantities ('leads') are score[p2] - score[k] for the
nine k != p2; the network is right when all nine are positive.

Background quadrature: the truncated normal on (-inf, m3) is integrated in probability space,
x = Phi^-1(q Phi(m3)), q on Gauss-Legendre nodes in (0, 1), equal-mass weights.
"""
import itertools, os
import numpy as np, torch
os.environ.setdefault("OMP_NUM_THREADS", "3"); torch.set_num_threads(3)
DEV = "cuda"; dt = torch.float64
torch.cuda.set_per_process_memory_fraction(11 / 32.6)   # shared GPU: stay under about 11 GB
SQ2 = np.sqrt(2.0)
TRIPLES = torch.tensor(list(itertools.permutations(range(10), 3)), device=DEV)


def ncdf(a): return 0.5 * (1 + torch.erf(a / SQ2))
def nicdf(p): return SQ2 * torch.erfinv(2 * p - 1)


def background_mean(m3):
    r = torch.exp(-m3 * m3 / 2) / np.sqrt(2 * np.pi) / ncdf(m3).clamp_min(1e-300)
    return -r


def bg_nodes(m3, Q):
    """Q quadrature nodes [n, Q] and weights [Q] (sum 1) for N(0,1) truncated below m3."""
    x, w = np.polynomial.legendre.leggauss(Q); q = torch.tensor(0.5 * (x + 1), device=DEV, dtype=dt)
    w = torch.tensor(0.5 * w, device=DEV, dtype=dt)
    p = (q[None, :] * ncdf(m3)[:, None]).clamp(1e-300, 1 - 1e-16)
    return nicdf(p), w


FAST = torch.float32   # network arithmetic precision (fp64 is very slow on this GPU); results returned in float64


def run(W, X):
    wi, wh, wo = (w.to(FAST) for w in W); Xf = X.to(FAST); h = torch.zeros(X.shape[0], wi.shape[0], device=DEV, dtype=FAST)
    for t in range(10): h = torch.relu(h @ wh.T + Xf[:, t:t + 1] * wi)
    return (h @ wo.T).to(X.dtype)


def run_jac(W, X):
    """Scores [n,10] and exact local linear map G [n, 10 inputs, 10 scores] (on/off pattern of X held)."""
    wi, wh, wo = (w.to(FAST) for w in W); n = X.shape[0]; d = wi.shape[0]; Xf = X.to(FAST)
    h = torch.zeros(n, d, device=DEV, dtype=FAST); J = torch.zeros(n, 10, d, device=DEV, dtype=FAST)
    for t in range(10):
        a = h @ wh.T + Xf[:, t:t + 1] * wi; on = (a > 0).to(FAST)
        J = J @ wh.T; J[:, t, :] += wi; J = J * on[:, None, :]; h = a * on
    return (h @ wo.T).to(X.dtype), (J @ wo.T).to(X.dtype)


def lead_matrix(p2):
    """A [n, 9, 10] with leads = A @ scores: score[p2] - score[k], k != p2 in increasing order."""
    n = p2.shape[0]
    others = torch.stack([torch.tensor([k for k in range(10) if k != j]) for j in range(10)]).to(DEV)[p2]
    A = torch.zeros(n, 9, 10, device=DEV, dtype=dt)
    A[torch.arange(n)[:, None], torch.arange(9, device=DEV)[None], p2[:, None].expand(n, 9)] = 1.0
    A.scatter_(2, others[:, :, None], -1.0)
    return A, others


def base_inputs(pos, val):
    n = pos.shape[0]; X = background_mean(val[:, 2])[:, None].repeat(1, 10); ar = torch.arange(n, device=DEV)
    isbg = torch.ones(n, 10, dtype=torch.bool, device=DEV)
    for k in range(3): X[ar, pos[:, k]] = val[:, k]; isbg[ar, pos[:, k]] = False
    return X, isbg


_Z = {}
def orthant(lm, lc, nq=2048):
    """P(all leads > 0) for leads ~ N(lm, lc), [n,9], [n,9,9]; fixed scrambled Sobol normal points."""
    if nq not in _Z:
        e = torch.quasirandom.SobolEngine(9, scramble=True, seed=7).draw(nq).to(DEV, dt)
        _Z[nq] = SQ2 * torch.erfinv(2 * e.clamp(1e-9, 1 - 1e-9) - 1)
    I = torch.eye(9, device=DEV, dtype=dt)
    Lc = torch.linalg.cholesky(lc + I * (1e-9 + 1e-9 * torch.diagonal(lc, dim1=-2, dim2=-1).mean(1))[:, None, None])
    leads = lm[:, None, :] + _Z[nq][None] @ Lc.transpose(1, 2)
    return (leads > 0).all(2).to(dt).mean(1)


def orthant_cheap(lm, lc, k=3):
    """Same, but only the k smallest standardized leads joint (others treated as independent factors).
    Exact for k=9 up to quadrature; used for speed."""
    sd = torch.diagonal(lc, dim1=-2, dim2=-1).clamp_min(1e-30).sqrt(); z = lm / sd
    idx = z.argsort(1)[:, :k]
    lm2 = torch.gather(lm, 1, idx); lc2 = torch.gather(torch.gather(lc, 1, idx[:, :, None].expand(-1, -1, 9)), 2, idx[:, None, :].expand(-1, k, -1))
    rest = torch.ones_like(z).scatter(1, idx, 0.0)
    pr = torch.where(rest > 0, ncdf(z), torch.ones_like(z)).prod(1)
    return pr * orthant_k(lm2, lc2)


_Zk = {}
def orthant_k(lm, lc, nq=1024):
    k = lm.shape[1]
    if k not in _Zk:
        e = torch.quasirandom.SobolEngine(k, scramble=True, seed=7).draw(nq).to(DEV, dt)
        _Zk[k] = SQ2 * torch.erfinv(2 * e.clamp(1e-9, 1 - 1e-9) - 1)
    I = torch.eye(k, device=DEV, dtype=dt)
    Lc = torch.linalg.cholesky(lc + I * (1e-9 + 1e-9 * torch.diagonal(lc, dim1=-2, dim2=-1).mean(1))[:, None, None])
    return ((lm[:, None, :] + _Zk[k][None] @ Lc.transpose(1, 2)) > 0).all(2).to(dt).mean(1)


def load_net(name):
    import sys; sys.path.insert(0, "/data/output/research/algzoo/20261017_handbuilt/family")
    from algzoo.core import load, example
    if name == "ARC": p = example()
    else: p = load(f"/data/output/research/algzoo/20261016_run3/models/2nd_argmax_16_10_1073741824_{name}.pth")
    return tuple(torch.tensor(np.asarray(q, float), device=DEV, dtype=dt) for q in p)


def noisy_copies(n=200):
    """Regenerate the 200 noisy copies of hilton_noise.py exactly (same generator, same draw order)."""
    from algzoo.core import example
    g = torch.Generator(device=DEV).manual_seed(41); N = 100000
    torch.randn(N, 10, device=DEV, generator=g, dtype=dt)
    W = [torch.as_tensor(np.asarray(w, float), dtype=dt, device=DEV) for w in example()]
    shapes = [w.shape for w in W]; sizes = [w.numel() for w in W]
    dws = [torch.randn(sum(sizes), device=DEV, dtype=dt, generator=g) * 0.0025 for _ in range(n)]
    out = []
    for v in dws:
        i = 0; P = []
        for s, sh, w in zip(sizes, shapes, W): P.append(w + v[i:i + s].reshape(sh)); i += s
        out.append(tuple(P))
    return out
