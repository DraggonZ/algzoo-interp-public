"""Shared setup for span tests (imported): 50,000 lists, their nine leads and weight gradients, clean states, level weightings by K.
See span_test.py for the method."""
import numpy as np, torch as th
from torch.func import vmap, grad
from noiselib import net, DEV, DT

W = net("ARC"); P = 432; C = np.load("copies.npz")
dws = th.as_tensor(C["ARC_0.0025_dw"], device=DEV); y = C["ARC_0.0025_acc"]
g = th.Generator(device=DEV).manual_seed(43); N = 50000
X = th.randn(N, 10, device=DEV, dtype=DT, generator=g); yy = X.argsort(-1)[:, -2]
def logits(params, x):
    wi, wh, wo = params; h = th.zeros(16, dtype=x.dtype, device=x.device)
    for t in range(10): h = th.relu(wh @ h + wi * x[t])
    return wo @ h
jac = vmap(grad(lambda p, x, e: (logits(p, x) * e).sum()), in_dims=(None, 0, 0))
with th.no_grad(): L = vmap(logits, in_dims=(None, 0))(W, X)
oy = th.nn.functional.one_hot(yy, 10).double(); keep = th.arange(10, device=DEV).unsqueeze(0) != yy.unsqueeze(1)
D9 = ((L * oy).sum(1, keepdim=True) - L)[keep].reshape(-1, 9)
G = th.zeros(N, 10, P, device=DEV, dtype=th.float32)
for b in range(0, N, 5000):
    sl = slice(b, b + 5000)
    for k in range(10):
        G[sl, k] = th.cat([gi.reshape(5000, -1) for gi in jac(W, X[sl], oy[sl] - th.nn.functional.one_hot(th.full((5000,), k, device=DEV), 10).double())], 1).float()
G9 = G[keep].reshape(N, 9, P); del G

def propagate(A):
    """A: [k, 432] quantity gradients. Per-copy first-order accuracy with the noise projected on span(A)."""
    Q, _ = th.linalg.qr(A.T.double()); Qf = Q.float(); GQ = G9 @ Qf
    est = np.array([((D9 + (GQ @ (Qf.T @ v.float())).double()) > 0).all(1).double().mean().item() for v in dws])
    return est
def rep(p): return f"RMSE {100 * np.sqrt(np.mean((p - y) ** 2)):.2f} points, correlation {np.corrcoef(p, y)[0, 1]:.3f}"

# clean states and K
g2 = th.Generator(device=DEV).manual_seed(7); n = 200000
Xs = th.randn(n, 10, device=DEV, dtype=DT, generator=g2)
wi, wh, wo = W; h = th.zeros(n, 16, device=DEV, dtype=DT); Hp = []
for t in range(10): Hp.append(h); h = th.relu(h @ wh.T + Xs[:, t:t + 1] * wi)
Hp = th.stack(Hp, 1)                                                           # state before x_t
cm = th.cummax(Xs, 1); m1 = th.cat([th.zeros(n, 1, device=DEV, dtype=DT), cm.values[:, :-1]], 1).clamp(min=0)
pos = th.cat([th.full((n, 1), -1, device=DEV), cm.indices[:, :-1]], 1)
K = th.arange(10, device=DEV)[None] - pos                                      # steps since the largest (before x_t)
srt = []                                                                       # runner-up before x_t
m2 = th.zeros(n, 10, device=DEV, dtype=DT)
for t in range(2, 10): m2[:, t] = Xs[:, :t].topk(2, 1).values[:, 1].clamp(min=0)
ok = (m1 > 0.3) & (pos >= 0)
HIDE = [1, 6, 7, 8]; MEM = [0, 3, 10, 11, 12, 13, 14, 15]
BINS = {"K=1": [1], "K=2-3": [2, 3], "K=4-6": [4, 5, 6], "K=7-9": [7, 8, 9]}
def row_vec(nrn, vec, scale=1.0):
    a = th.zeros(P, device=DEV, dtype=DT); a[16 + 16 * nrn:16 + 16 * (nrn + 1)] = vec / scale; return a
lvl, split = {}, {}
for b, ks in BINS.items():
    m = ok & th.isin(K, th.tensor(ks, device=DEV))
    lvl[b] = (Hp[m] / m1[m][:, None]).mean(0)
    Z = th.stack([m1[m], m2[m]], 1); coef = th.linalg.lstsq(Z, Hp[m]).solution    # [2, 16]
    split[b] = coef
allK = (Hp[ok] / m1[ok][:, None]).mean(0)

