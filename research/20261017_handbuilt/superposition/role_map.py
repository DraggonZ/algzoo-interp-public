"""First test of the 'wide hand-built network as a dictionary' idea. Each trained neuron's pre-activation at every step
(ARC's network) is fitted as one linear map, shared over steps, from a candidate network's activations at the same step
plus the current number. Fit on 100,000 random lists, scored on 100,000 others: R^2 of the pre-activation, R^2 of the
activation after a ReLU of the fit, and how many candidate neurons greedy forward selection needs for 90% of the full
fit's R^2 (a first measure of how many roles a trained neuron mixes). Candidates: v21 (28 roles), v18 (16), three
random 28-neuron ReLU networks, and plain features (the ten numbers seen so far, zero for future ones; the running
largest and second-largest)."""
import sys, numpy as np, torch
sys.path.insert(0, "../family")
from common import *
from algzoo.handbuilt_v18 import build_v18
from algzoo.handbuilt_v21 import build_v21
def states(W, X):
    h = torch.zeros(len(X), len(W[0]), device=DEV, dtype=dt); H, Z = [], []
    for t in range(10):
        z = h @ W[1].T + X[:, t:t + 1] * W[0]; h = torch.relu(z); H.append(h); Z.append(z)
    return torch.stack(H, 1), torch.stack(Z, 1)     # [n, t, d]
def plain(X):
    n = len(X); F = []; m1 = torch.zeros(n, device=DEV, dtype=dt); m2 = torch.zeros(n, device=DEV, dtype=dt)
    for t in range(10):
        x = X[:, t]; m2 = torch.where(x > m1, m1, torch.maximum(m2, x)); m1 = torch.maximum(m1, x)
        past = torch.where(torch.arange(10, device=DEV)[None] <= t, X, 0)
        F.append(torch.cat([past, m1[:, None], m2[:, None]], 1))
    return torch.stack(F, 1)
def random_net(seed, d=28):
    g = np.random.default_rng(seed); wh = g.standard_normal((d, d)) / np.sqrt(d); wh *= 0.95 / np.abs(np.linalg.eigvals(wh)).max()
    return T([g.standard_normal(d), wh, np.zeros((10, d))])
def fit_eval(F, Ztr, Zte, Ftr_idx, Fte_idx, Htr_te):
    pass
if __name__ == "__main__":
    g = torch.Generator(device=DEV).manual_seed(77); X = torch.randn(200000, 10, device=DEV, dtype=dt, generator=g)
    Ht, Zt = states(ARC, X)
    cands = {"v21 (28 roles)": states(T(build_v21()), X)[0], "v18 (16)": states(T(build_v18().params), X)[0]}
    for s in range(3): cands[f"random 28, seed {s}"] = states(random_net(s), X)[0]
    cands["plain features (12)"] = plain(X)
    tr, te = slice(0, 100000), slice(100000, None)
    print(f"{'candidate':22s} | R2 pre-activation by trained neuron (0-15) | mean R2 pre / after ReLU | roles for 90%: by neuron")
    for nm, F in cands.items():
        F = torch.cat([F, X[:, :, None]], 2)                 # plus the current number
        F = F / (F[tr].reshape(-1, F.shape[2]).std(0) + 1e-12)
        A = torch.cat([F, torch.ones_like(F[..., :1])], 2)   # intercept (the trained network has none, the fit may)
        Atr = A[tr].reshape(-1, A.shape[2]); Ate = A[te].reshape(-1, A.shape[2])
        G = Atr.T @ Atr; G += 1e-6 * torch.trace(G) / len(G) * torch.eye(len(G), device=DEV, dtype=dt)
        r2p, r2a, nroles = [], [], []
        for j in range(16):
            y = Zt[tr][..., j].reshape(-1); yt = Zt[te][..., j].reshape(-1); at = Ht[te][..., j].reshape(-1)
            w = torch.linalg.solve(G, Atr.T @ y); p = Ate @ w
            r2 = 1 - ((p - yt) ** 2).mean() / yt.var(); r2p.append(r2.item())
            r2a.append((1 - ((torch.relu(p) - at) ** 2).mean() / at.var()).item())
            # greedy forward selection on a subsample (always with intercept)
            sub = torch.arange(0, len(Atr), 7, device=DEV); As, ys = Atr[sub], y[sub]; chosen = [A.shape[2] - 1]; best = 0.0
            for k in range(A.shape[2] - 1):
                cand_best, cb = -1, None
                for c in range(A.shape[2] - 1):
                    if c in chosen: continue
                    idx = chosen + [c]; M = As[:, idx]; ww = torch.linalg.lstsq(M, ys[:, None]).solution
                    rr = 1 - ((M @ ww)[:, 0] - ys).pow(2).mean() / ys.var()
                    if rr > cand_best: cand_best, cb = rr.item(), c
                chosen.append(cb)
                if cand_best >= 0.9 * max(r2.item(), 1e-9): break
            nroles.append(len(chosen) - 1)
        print(f"{nm:22s} | " + " ".join(f"{v:.2f}" for v in r2p) + f" | {np.mean(r2p):.3f} / {np.mean(r2a):.3f} | " + " ".join(str(k) for k in nroles))
