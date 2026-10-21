"""Sampling-free accuracy of ARC's network: the three largest numbers exact (every ordered position triple, values on a
grid), the other seven at the mean of a normal truncated below the third. acc = sum over triples of the integral of
phi(m1) phi(m2) phi(m3) Phi(m3)^7 [the second-largest's position wins], m1 > m2 > m3. Grid: m3 = u (Gauss-Legendre),
m2 = u + a, m1 = m2 + b, with a, b = R s^2 (midpoints in s, dense near ties). Run at two grid sizes to see convergence."""
import itertools, sys, time
import numpy as np, torch
import adf
DEV = "cuda"; dt = torch.float64
T = torch.tensor(list(itertools.permutations(range(10), 3)), device=DEV)


def grid(nu, na, R=5.0):
    x, w = np.polynomial.legendre.leggauss(nu); u = 0.5 * (x + 1) * 6.5 - 2.5; wu = 0.5 * 6.5 * w
    s = (np.arange(na) + 0.5) / na; a = R * s ** 2; wa = R * 2 * s / na
    U, A, B = np.meshgrid(u, a, a, indexing="ij"); WU, WA, WB = np.meshgrid(wu, wa, wa, indexing="ij")
    m3 = U.ravel(); m2 = m3 + A.ravel(); m1 = m2 + B.ravel()
    from scipy.stats import norm
    wt = WU.ravel() * WA.ravel() * WB.ravel() * norm.pdf(m1) * norm.pdf(m2) * norm.pdf(m3) * norm.cdf(m3) ** 7
    return np.c_[m1, m2, m3], wt


def win_hard(W, pos, val):
    wi, wh, wo = W; n = pos.shape[0]
    bmu, _ = adf.background(val[:, 2])
    X = bmu[:, None].repeat(1, 10); ar = torch.arange(n, device=DEV)
    for k in range(3): X[ar, pos[:, k]] = val[:, k]
    h = torch.zeros(n, wi.shape[0], device=DEV, dtype=dt)
    for t in range(10): h = torch.relu(h @ wh.T + X[:, t:t + 1] * wi)
    return ((h @ wo.T).argmax(1) == pos[:, 1]).to(dt)


def estimate(W, nu, na, chunk=1_500_000):
    V, wt = grid(nu, na); V = torch.tensor(V, device=DEV, dtype=dt); wt = torch.tensor(wt, device=DEV, dtype=dt)
    nV = len(wt); total = torch.zeros((), device=DEV, dtype=dt); mass = wt.sum() * len(T)
    per = max(1, chunk // nV)
    for i in range(0, len(T), per):
        tt = T[i:i + per]; pos = tt.repeat_interleave(nV, 0); val = V.repeat(len(tt), 1)
        total += (win_hard(W, pos, val) * wt.repeat(len(tt))).sum()
    return (total / mass).item(), mass.item()


if __name__ == "__main__":
    W = adf.net(None)
    for nu, na in [(24, 24), (40, 48), (56, 72)]:
        t0 = time.time(); acc, mass = estimate(W, nu, na)
        print(f"grid {nu} x {na} x {na} ({nu*na*na*720/1e6:.1f}M network runs on fixed inputs): estimate {100*acc:.2f}%  (probability mass {mass:.4f}, should be 1)  {time.time()-t0:.0f}s", flush=True)
