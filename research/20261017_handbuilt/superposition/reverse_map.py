"""Reverse direction: can each of v21's 28 roles be read off ARC's 16 trained neurons (one linear map shared over
steps, plus the current number, no intercept; held-out R^2)? Also the effective dimension of each network's state
(participation ratio of the covariance eigenvalues, pooled over steps) and the canonical correlations between ARC's
state and v21's (how many shared directions). Controls: v21 roles read from a random 16-neuron network, and from the
plain features."""
import sys, numpy as np, torch
sys.path.insert(0, "../family")
from common import *
from role_map import states, plain, random_net
from algzoo.handbuilt_v21 import build_v21, NAMES
def ridge_r2(Ftr, Fte, Ytr, Yte):
    G = Ftr.T @ Ftr; G += 1e-8 * torch.trace(G) / len(G) * torch.eye(len(G), device=DEV, dtype=dt)
    W = torch.linalg.solve(G, Ftr.T @ Ytr); P = Fte @ W
    return 1 - ((P - Yte) ** 2).mean(0) / Yte.var(0)
def pr(H):
    C = torch.cov(H.T); e = torch.linalg.eigvalsh(C).clamp(min=0); return (e.sum() ** 2 / (e ** 2).sum()).item(), e.flip(0)
if __name__ == "__main__":
    g = torch.Generator(device=DEV).manual_seed(78); X = torch.randn(200000, 10, device=DEV, dtype=dt, generator=g)
    tr, te = slice(0, 100000), slice(100000, None)
    R = states(T(build_v21()), X)[0]; Ha = states(ARC, X)[0]
    srcs = {"ARC (16)": Ha, "random 16, seed 0": states(random_net(0, 16), X)[0], "random 16, seed 1": states(random_net(1, 16), X)[0], "plain features (12)": plain(X)}
    flat = lambda A, s: torch.cat([A[s], X[s][:, :, None]], 2).reshape(-1, A.shape[2] + 1)
    Y = R / (R[tr].reshape(-1, 28).std(0) + 1e-12); Ytr, Yte = Y[tr].reshape(-1, 28), Y[te].reshape(-1, 28)
    print("v21 roles: " + " ".join(NAMES))
    for nm, S in srcs.items():
        sd = S[tr].reshape(-1, S.shape[2]).std(0) + 1e-12; S = S / sd
        r2 = ridge_r2(flat(S, tr), flat(S, te), Ytr, Yte)
        print(f"read from {nm:20s}: mean R2 {r2.mean():.3f}; roles above 0.9: {(r2 > 0.9).sum().item()}, above 0.99: {(r2 > 0.99).sum().item()} | " + " ".join(f"{v:.2f}" for v in r2.tolist()))
    for nm, S in (("ARC", Ha), ("v21", R)):
        p, e = pr(S.reshape(-1, S.shape[2])); print(f"{nm}: effective dimension (participation ratio) {p:.1f}; variance in top 4/8/12/16 directions: " + " ".join(f"{(e[:k].sum()/e.sum()).item():.3f}" for k in (4, 8, 12, 16)))
    # canonical correlations
    A = Ha.reshape(-1, 16); B = R.reshape(-1, 28); A = A - A.mean(0); B = B - B.mean(0)
    def whiten(M):
        C = M.T @ M / len(M); e, V = torch.linalg.eigh(C); keep = e > 1e-10 * e.max(); return M @ V[:, keep] / e[keep].sqrt()
    s = torch.linalg.svdvals(whiten(A).T @ whiten(B) / len(A))
    print("canonical correlations ARC vs v21: " + " ".join(f"{v:.3f}" for v in s.tolist()))
