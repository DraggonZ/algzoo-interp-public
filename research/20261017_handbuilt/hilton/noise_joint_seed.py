"""Second pass of noise_first_order.py: the nine leads D_k of the right answer over the other positions move together
under weight noise. To first order (ReLU pattern fixed) they are jointly normal with covariance
sigma^2 C, C_kl = sum over weights (w dD_k/dw)(w dD_l/dw). Predicted accuracy = mean over lists of P(D + noise > 0 in
all nine), computed per list by drawing 4,000 draws of that 9-dimensional normal (a Gaussian integral, no network runs).
Also the predicted spread of accuracy across copies is not attempted here."""
import numpy as np
import torch as th
from torch.func import vmap, grad
from algzoo.core import example
dev = "cuda"
W = [th.as_tensor(w, dtype=th.float64, device=dev) for w in example()]
g = th.Generator(device=dev).manual_seed(int(__import__("sys").argv[1]) if len(__import__("sys").argv) > 1 else 7); N = 60000
X = th.randn(N, 10, device=dev, generator=g, dtype=th.float64); y = X.argsort(-1)[:, -2]
def logits(params, x):
    wi, wh, wo = params; h = th.zeros(16, dtype=x.dtype, device=x.device)
    for t in range(10): h = th.relu(wh @ h + wi * x[t])
    return wo @ h
def diff(params, x, e): return (logits(params, x) * e).sum()
jac = vmap(grad(diff), in_dims=(None, 0, 0))
with th.no_grad(): L = vmap(logits, in_dims=(None, 0))(W, X)
oy = th.nn.functional.one_hot(y, 10).double()
C = th.zeros(N, 10, 10, device=dev, dtype=th.float64); D = (L * oy).sum(1, keepdim=True) - L
B = 10000
for b in range(0, N, B):
    sl = slice(b, b + B); Gs = []
    for k in range(10):
        E = oy[sl] - th.nn.functional.one_hot(th.full((len(X[sl]),), k, device=dev), 10).double()
        gr = jac(W, X[sl], E); Gs.append(th.cat([(gi * wi.unsqueeze(0)).reshape(len(gi), -1) for gi, wi in zip(gr, W)], 1))
    G = th.stack(Gs, 1)                                  # [B, 10, 432]
    C[sl] = G @ G.transpose(1, 2)
idx = th.arange(10, device=dev)
for sig, true in ((0.005, 0.908), (0.01, 0.813), (0.02, 0.643)):
    res = []
    for b in range(0, N, B):
        sl = slice(b, b + B); yb = y[sl]
        keep = idx.unsqueeze(0) != yb.unsqueeze(1)       # nine competitors
        Db = D[sl][keep].reshape(-1, 9)
        Cb = C[sl][keep.unsqueeze(2) & keep.unsqueeze(1)].reshape(-1, 9, 9) * sig ** 2
        Lc = th.linalg.cholesky(Cb + 1e-12 * th.eye(9, device=dev, dtype=th.float64))
        Z = th.randn(len(Db), 4000, 9, device=dev, dtype=th.float64, generator=g)
        ok = ((Db.unsqueeze(1) + Z @ Lc.transpose(1, 2)) > 0).all(2).double().mean(1)
        res.append(ok)
    print(f"sigma {sig}: true {true:.3f}; predicted (joint, first order) {th.cat(res).mean().item():.3f}")
