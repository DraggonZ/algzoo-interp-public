"""E4: deterministic estimate for one linear map per arrangement. For each of the 720 arrangements (p1, p2, p3):
on/off from one canonical run, map M (scores = M x), margins d_j = (M[p2] - M[j]) x for j != p2. Quadrature over
u = m3 (Gauss-Legendre on [-3, 3.5]) and gaps a = m2 - m3, b = m1 - m2 (Gauss-Legendre on [0, 4] in sqrt scale),
density phi(m1) phi(m2) phi(m3) Phi(m3)^7. Smaller numbers: Gaussian with the truncated-normal mean and variance.
P(right | m) = BVN(margins over p1 and p3) * prod over smaller positions of P(margin > 0).
Also the surrogate's own accuracy on 1M sampled lists (diagnostic) and the integral's grid convergence."""
import itertools, sys, time
from elib import *
PERM = torch.tensor(list(itertools.permutations(range(10), 3)), device=DEV)
N01 = torch.distributions.Normal(torch.tensor(0., device=DEV, dtype=DT), torch.tensor(1., device=DEV, dtype=DT))
Phi = N01.cdf; phi = lambda z: torch.exp(-z * z / 2) / np.sqrt(2 * np.pi)
gl = lambda n: [torch.tensor(a, device=DEV, dtype=DT) for a in np.polynomial.legendre.leggauss(n)]
R10, WR10 = gl(12)
Q, WQ = gl(24); Q = (Q + 1) / 2; WQ = WQ / 2

def bvn(h, k, rho):
    """P(X > -h, Y > -k) for standard bivariate normal with correlation rho = Phi2(h, k; rho), Plackett's formula."""
    r = rho[..., None] * (R10 + 1) / 2; w = rho[..., None] * WR10 / 2
    hh, kk = h[..., None], k[..., None]; s = torch.sqrt((1 - r * r).clamp_min(1e-12))
    f = torch.exp(-(hh * hh - 2 * r * hh * kk + kk * kk) / (2 * s * s)) / (2 * np.pi * s)
    return Phi(h) * Phi(k) + (w * f).sum(-1)

def maps(W):
    """Canonical-run on/off per arrangement and the exact linear map M [720, 10, 10] (scores = M x)."""
    Xc = torch.full((720, 10), -0.43, device=DEV, dtype=DT); ar = torch.arange(720, device=DEV)
    for k, v in enumerate((1.539, 1.001, 0.656)): Xc[ar, PERM[:, k]] = v
    _, on = run(W, Xc); wi, wh, wo = W
    G = torch.zeros(720, len(wi), 10, device=DEV, dtype=DT)          # state as linear map of the inputs
    for t in range(10):
        G = wh @ G; G[:, :, t] += wi; G = G * on[:, t, :, None]
    return wo @ G, on

def estimate(M, nu, na):
    u, wu = gl(nu); lo, hi = -3.0, 3.5; u = lo + (hi - lo) * (u + 1) / 2; wu = wu * (hi - lo) / 2
    s, ws = gl(na); s = 2.0 * (s + 1) / 2; ws = ws * 2.0 / 2; g = s * s; wg = ws * 2 * s      # gap = s^2, s in [0, 2]
    U, A, B = torch.meshgrid(u, g, g, indexing="ij"); Wq = (wu[:, None, None] * wg[None, :, None] * wg[None, None, :]).reshape(-1)
    m3, m2 = U.reshape(-1), (U + A).reshape(-1); m1 = m2 + B.reshape(-1)
    dens = phi(m1) * phi(m2) * phi(m3) * Phi(m3) ** 7
    lam = phi(m3) / Phi(m3).clamp_min(1e-300); mu = -lam; var = 1 - m3 * lam - lam * lam      # truncated below m3
    tot = torch.zeros((), device=DEV, dtype=DT); parts = torch.zeros(3, device=DEV, dtype=DT)
    for i in range(720):
        p1, p2, p3 = PERM[i].tolist(); bg = [k for k in range(10) if k not in (p1, p2, p3)]
        C = M[i, p2][None, :] - M[i]                                    # [10 competitors, 10 inputs]
        Cb = C[:, bg]                                                    # [10, 7]
        mean = C[:, p1][:, None] * m1 + C[:, p2][:, None] * m2 + C[:, p3][:, None] * m3 + Cb.sum(1)[:, None] * mu   # [10, P]
        cov = (Cb @ Cb.T)[:, :, None] * var                              # [10, 10, P]
        sd = cov.diagonal(0, 0, 1).T.clamp_min(1e-30).sqrt()             # [10, P]
        z = mean / sd
        rho = cov[p1, p3] / (sd[p1] * sd[p3])
        main = bvn(z[p1], z[p3], rho.clamp(-0.999999, 0.999999))
        # margin over a smaller position k: x_k exact (truncated below m3), the other six Gaussian
        ck = C[bg, bg].diagonal() if False else torch.stack([C[k, k] for k in bg])            # coefficient of x_k in its own margin
        mrest = mean[bg] - ck[:, None] * mu                                                    # [7, P]
        vrest = (cov.diagonal(0, 0, 1).T[bg] - ck[:, None] ** 2 * var).clamp_min(1e-30)
        xk = torch.special.ndtri((Q[:, None] * Phi(m3)[None, :]).clamp(1e-300, 1 - 1e-16))      # [nq, P] quantiles
        pk = (WQ[None, :, None] * Phi((mrest[:, None, :] + ck[:, None, None] * xk[None]) / vrest.sqrt()[:, None, :])).sum(1)
        others = torch.prod(pk, 0)
        pr = main * others
        tot += (pr * dens * Wq).sum()
        parts += torch.stack([((1 - Phi(z[p1])) * dens * Wq).sum(), ((1 - Phi(z[p3])) * dens * Wq).sum(), ((1 - others) * dens * Wq).sum()])
    return 100 * tot.item(), 100 * parts.cpu().numpy(), len(m1)

if __name__ == "__main__":
    W = arc(); M, on = maps(W)
    X = lists(1_000_000, 102)
    o = X.argsort(1, descending=True)[:, :3]; LUT = torch.full((1000,), -1, dtype=torch.long, device=DEV)
    LUT[(PERM[:, 0] * 10 + PERM[:, 1]) * 10 + PERM[:, 2]] = torch.arange(720, device=DEV)
    ai = LUT[(o[:, 0] * 10 + o[:, 1]) * 10 + o[:, 2]]
    S = torch.einsum("nij,nj->ni", M[ai], X); y = o[:, 1]; w = S.argmax(1)
    print(f"surrogate on 1M sampled lists (diagnostic): {acc(S, X):.2f}; mistakes naming the largest {100 * (w == o[:, 0]).double().mean():.2f}, "
          f"the third {100 * (w == o[:, 2]).double().mean():.2f}, another {100 * ((w != y) & (w != o[:, 0]) & (w != o[:, 2])).double().mean():.2f}")
    for nu, na in ((12, 10), (24, 20), (48, 40), (96, 64)):
        t0 = time.time(); e, p, npts = estimate(M, nu, na)
        print(f"grid {nu} x {na} x {na} ({npts:,} points x 720): estimate {e:.2f}; single-margin losses to the largest {p[0]:.2f}, "
              f"the third {p[1]:.2f}, the smaller seven {p[2]:.2f}; {time.time() - t0:.1f} s", flush=True)
