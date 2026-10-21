"""At a 2nd-3rd tie (positions and values of the three largest fixed), how do the seven smaller numbers move the lean
(score of the true second's position minus the third's, at the tie) and the wrong width (exact walk)? Compare the
spread over random smaller numbers with (a) the value at their mean, (b) a Gaussian from the gradient at the mean."""
import sys, numpy as np, torch
sys.path.insert(0, "../../../family")
from common import ARC, trained, scores, jac
from linelib import *
from scipy.stats import truncnorm
Phi = lambda x: torch.special.ndtr(x)
def tn(n, T, g):   # n x 7 truncated normals below T
    u = torch.rand(n, 7, device=DEV, dtype=dt, generator=g) * Phi(torch.tensor(T, dtype=dt))
    return torch.special.ndtri(u)
torch.set_printoptions(precision=3)
for name, W in (("ARC", ARC), ("114", trained(114))):
    g = torch.Generator(device=DEV).manual_seed(0); rng = np.random.default_rng(0)
    print(name)
    for trial in range(8):
        pL, pS, pT = rng.choice(10, 3, replace=False); L, T = 1.6, 0.8
        rest = [i for i in range(10) if i not in (pL, pS, pT)]
        n = 4000; Z = tn(n, T, g); X = torch.zeros(n, 10, device=DEV, dtype=dt); X[:, rest] = Z; X[:, pL] = L; X[:, pS] = T; X[:, pT] = T
        s = scores(W, X); lam = s[:, pS] - s[:, pT]
        # wrong width along S from T: exact walk to where pS wins
        k = torch.full((n,), int(pS), device=DEV); (r, lo, hi), _, _ = walk(W, X, k, k, torch.full((n,), L - T, device=DEV, dtype=dt))
        right = torch.zeros(n, device=DEV, dtype=dt).index_add_(0, r, Phi(T + hi) - Phi(T + lo))
        wrong = (Phi(torch.tensor(L, dtype=dt)) - Phi(torch.tensor(T, dtype=dt))).item() - right
        zm = float(-np.exp(-T * T / 2) / np.sqrt(2 * np.pi) / Phi(torch.tensor(T, dtype=dt)).item())   # mean of N truncated below T
        zv = float(1 - T * (-zm) - zm ** 2)   # var: 1 - T*phi/Phi - (phi/Phi)^2
        Xm = X[:1].clone(); Xm[:, rest] = zm; sm = scores(W, Xm); lam0 = (sm[0, pS] - sm[0, pT]).item()
        J = jac(W, Xm)[0]; gl = (J[pS] - J[pT])[rest]; sd_lin = (gl.pow(2).sum() * zv).sqrt().item()
        km = None
        print(f"  L@{pL} S@{pS} T@{pT}: lean mean {lam.mean():+.3f} sd {lam.std():.3f} | at mean {lam0:+.3f}, linear sd {sd_lin:.3f} | wrong mass mean {1e3*wrong.mean():.2f}e-3 sd {1e3*wrong.std():.2f}e-3, at mean z {1e3*wrong[0]*0:.0f}")
