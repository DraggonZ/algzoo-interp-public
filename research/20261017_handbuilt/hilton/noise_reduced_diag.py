"""Why does the three-input reduction overpredict accuracy under weight noise (noise_joint_reduced.log: 0.916, 0.835,
0.682 against true 0.908, 0.813, 0.643)? On the same 40,000 lists compute, for full and for reduced inputs, the nine
leads D and their weight-noise covariance C; then predict with the four combinations (D full or reduced) x (C full or
reduced). Also: the leads' median and the median noise size per lead in each case."""
import numpy as np
import torch as th
from torch.func import vmap, grad
from algzoo.core import example
dev = "cuda"
W = [th.as_tensor(w, dtype=th.float64, device=dev) for w in example()]
g = th.Generator(device=dev).manual_seed(7); N = 40000
X = th.randn(N, 10, device=dev, generator=g, dtype=th.float64); y = X.argsort(-1)[:, -2]
top = X.argsort(-1)[:, -3:]; Xr = th.full_like(X, -0.46); Xr.scatter_(1, top, X.gather(1, top))
def logits(params, x):
    wi, wh, wo = params; h = th.zeros(16, dtype=x.dtype, device=x.device)
    for t in range(10): h = th.relu(wh @ h + wi * x[t])
    return wo @ h
def diff(params, x, e): return (logits(params, x) * e).sum()
jac = vmap(grad(diff), in_dims=(None, 0, 0))
oy = th.nn.functional.one_hot(y, 10).double(); idx = th.arange(10, device=dev)
keep = idx.unsqueeze(0) != y.unsqueeze(1)
def DC(Xin):
    with th.no_grad(): L = vmap(logits, in_dims=(None, 0))(W, Xin)
    D = (L * oy).sum(1, keepdim=True) - L; C = th.zeros(N, 10, 10, device=dev, dtype=th.float64)
    for b in range(0, N, 10000):
        sl = slice(b, b + 10000); Gs = []
        for k in range(10):
            E = oy[sl] - th.nn.functional.one_hot(th.full((10000,), k, device=dev), 10).double()
            gr = jac(W, Xin[sl], E); Gs.append(th.cat([(gi * wi.unsqueeze(0)).reshape(len(gi), -1) for gi, wi in zip(gr, W)], 1))
        G = th.stack(Gs, 1); C[sl] = G @ G.transpose(1, 2)
    return D[keep].reshape(-1, 9), C[keep.unsqueeze(2) & keep.unsqueeze(1)].reshape(-1, 9, 9)
Df, Cf = DC(X); Dr, Cr = DC(Xr)
def pred(D, C, sig):
    out = []
    for b in range(0, N, 10000):
        sl = slice(b, b + 10000)
        Lc = th.linalg.cholesky(C[sl] * sig ** 2 + 1e-12 * th.eye(9, device=dev, dtype=th.float64))
        Z = th.randn(10000, 3000, 9, device=dev, dtype=th.float64, generator=g)
        out.append(((D[sl].unsqueeze(1) + Z @ Lc.transpose(1, 2)) > 0).all(2).double().mean(1))
    return th.cat(out).mean().item()
mn = lambda D: D.min(1).values
print(f"median smallest lead: full {mn(Df).median().item():.2f}, reduced {mn(Dr).median().item():.2f}")
ru = lambda D: D.argmin(1)
sz = lambda D, C: C[th.arange(N), ru(D), ru(D)].sqrt().median().item()
print(f"median noise size of the smallest lead per unit sigma: full {sz(Df, Cf):.1f}, reduced {sz(Dr, Cr):.1f}")
for sig, true in ((0.005, 0.908), (0.01, 0.813), (0.02, 0.643)):
    print(f"sigma {sig}: true {true:.3f} | D full C full {pred(Df, Cf, sig):.3f} | D reduced C reduced {pred(Dr, Cr, sig):.3f} | "
          f"D reduced C full {pred(Dr, Cf, sig):.3f} | D full C reduced {pred(Df, Cr, sig):.3f}")
