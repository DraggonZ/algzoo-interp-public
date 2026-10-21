"""Hilton's noise (2026-10-18): every weight plus N(0, 0.0025^2), additive. Truth: 200 copies,
each scored on the same 100,000 lists (GPU). Predictions from the original network only:
 (a) mean accuracy: the nine leads jointly normal, covariance sigma^2 G G^T with G the leads' gradients wrt the 432
     weights (one backward pass per lead and list, ReLU pattern fixed);
 (b) each copy's accuracy: the leads moved by G dw for that copy's own noise dw (first order, no run of the copy).
Reported: true mean and spread of copies, (a), and for (b) the per-copy error (RMSE) and correlation with the truth."""
import numpy as np
import torch as th
from torch.func import vmap, grad
from algzoo.core import example
dev = "cuda"; SIG = 0.0025
g = th.Generator(device=dev).manual_seed(41); N = 100000
X = th.randn(N, 10, device=dev, generator=g, dtype=th.float64); y = X.argsort(-1)[:, -2]
W = [th.as_tensor(np.asarray(w, float), dtype=th.float64, device=dev) for w in example()]
shapes = [w.shape for w in W]; sizes = [w.numel() for w in W]
def batch_acc(P):
    wi, wh, wo = P; h = th.zeros(N, 16, dtype=th.float64, device=dev)
    for t in range(10): h = th.relu(h @ wh.T + X[:, t:t + 1] * wi)
    return ((h @ wo.T).argmax(1) == y).double().mean().item()
dws = [th.randn(sum(sizes), device=dev, dtype=th.float64, generator=g) * SIG for _ in range(200)]
def split(v):
    out = []; i = 0
    for s, sh in zip(sizes, shapes): out.append(v[i:i + s].reshape(sh)); i += s
    return out
true = np.array([batch_acc([w + d for w, d in zip(W, split(v))]) for v in dws])
print(f"original {batch_acc(W):.4f}; 200 copies: mean {true.mean():.4f}, SD {true.std():.4f}, range {true.min():.3f}-{true.max():.3f}")
def logits(params, x):
    wi, wh, wo = params; h = th.zeros(16, dtype=x.dtype, device=x.device)
    for t in range(10): h = th.relu(wh @ h + wi * x[t])
    return wo @ h
jac = vmap(grad(lambda p, x, e: (logits(p, x) * e).sum()), in_dims=(None, 0, 0))
with th.no_grad(): L = vmap(logits, in_dims=(None, 0))(W, X)
oy = th.nn.functional.one_hot(y, 10).double(); keep = th.arange(10, device=dev).unsqueeze(0) != y.unsqueeze(1)
D = ((L * oy).sum(1, keepdim=True) - L)
G = th.zeros(N, 10, sum(sizes), device=dev, dtype=th.float32)
for b in range(0, N, 5000):
    sl = slice(b, b + 5000)
    for k in range(10):
        gr = jac(W, X[sl], oy[sl] - th.nn.functional.one_hot(th.full((5000,), k, device=dev), 10).double())
        G[sl, k] = th.cat([gi.reshape(5000, -1) for gi in gr], 1).float()
D9 = D[keep].reshape(-1, 9); G9 = G[keep].reshape(N, 9, -1)
# (a) mean accuracy, leads jointly normal
pr = []
for b in range(0, N, 5000):
    Gb = G9[b:b + 5000].double(); C = SIG ** 2 * Gb @ Gb.transpose(1, 2)
    Lc = th.linalg.cholesky(C + 1e-12 * th.eye(9, device=dev, dtype=th.float64))
    Z = th.randn(len(Gb), 2000, 9, device=dev, dtype=th.float64, generator=g)
    pr.append(((D9[b:b + 5000].unsqueeze(1) + Z @ Lc.transpose(1, 2)) > 0).all(2).double().mean(1))
print(f"(a) predicted mean accuracy {th.cat(pr).mean().item():.4f} (true mean {true.mean():.4f})")
# (b) per copy, first order
est = []
for v in dws:
    lin = (G9 @ v.float()).double()                          # [N, 9]
    est.append(((D9 + lin) > 0).all(1).double().mean().item())
est = np.array(est); e = est - true
print(f"(b) per-copy first order: RMSE {100 * np.sqrt(np.mean(e ** 2)):.2f} points, bias {100 * e.mean():+.2f}, correlation {np.corrcoef(est, true)[0, 1]:.3f}")
print(f"    a constant guess (the true mean): RMSE {100 * true.std():.2f} points")
np.savez("hilton_noise.npz", true=true, est=est)
# (c) only lists near a tie need the linearization: a list whose smallest lead exceeds k x its leads' noise size
# (sigma x gradient norm) cannot change sign to first order at plausible draws; count it as it is.
size = SIG * G9.norm(dim=2).double()                           # [N, 9]
for k in (4, 6, 8):
    near = ((D9.abs() < k * size).any(1))
    base = ((D9 > 0).all(1) & ~near).double().sum()
    est_c = np.array([((base + ((D9[near] + (G9[near] @ v.float()).double()) > 0).all(1).double().sum()) / N).item() for v in dws])
    e = est_c - true
    print(f"(c) linearize only lists within {k} noise sizes of a tie: {near.double().mean().item():.3f} of lists; per-copy RMSE {100 * np.sqrt(np.mean(e ** 2)):.2f} points, correlation {np.corrcoef(est_c, true)[0, 1]:.3f}")
# (d) which weights carry the loss? per copy, the first-order accuracy with noise kept only in one block of weights
grp = {"store, detector (2, 4)": [2, 4], "gates (5, 9)": [5, 9], "hiding (1, 6, 7, 8)": [1, 6, 7, 8], "memory (0, 3, 10-15)": [0, 3, 10, 11, 12, 13, 14, 15]}
nwi, nwh = 16, 256
def mask_for(kind, rows=None, cols=None):
    m = th.zeros(sum(sizes), device=dev)
    if kind == "input": m[rows] = 1
    if kind == "rec":
        M = th.zeros(16, 16, device=dev); M[np.ix_(rows, cols)] = 1; m[nwi:nwi + nwh] = M.flatten()
    if kind == "out":
        M = th.zeros(10, 16, device=dev); M[:, cols] = 1; m[nwi + nwh:] = M.flatten()
    return m
allc = list(range(16)); base_loss = true.mean()
print("(d) mean first-order accuracy loss (points) with noise only in one block; full noise:", f"{100 * (0.953 - est.mean()):.2f}")
rows = []
for gn, gi in grp.items():
    for kind, m in (("input weights of", mask_for("input", rows=gi)), ("recurrent weights into", mask_for("rec", rows=gi, cols=allc)), ("readout weights of", mask_for("out", cols=gi))):
        acc_b = np.mean([((D9 + (G9 @ (v * m).float()).double()) > 0).all(1).double().mean().item() for v in dws[:60]])
        rows.append((100 * (0.953 - acc_b), f"{kind} {gn}", int(m.sum().item())))
for loss, name, n in sorted(rows, reverse=True):
    print(f"    {name:44s} ({n:3d} weights): {loss:5.2f}")
# (e) within the weights into the hiding neurons: by source
print("(e) noise only on the weights into the hiding neurons from one source group:")
for gn, gi in grp.items():
    m = mask_for("rec", rows=grp["hiding (1, 6, 7, 8)"], cols=gi)
    acc_b = np.mean([((D9 + (G9 @ (v * m).float()).double()) > 0).all(1).double().mean().item() for v in dws[:60]])
    print(f"    from {gn:24s} ({int(m.sum().item()):2d} weights): {100 * (0.953 - acc_b):5.2f}")
for src in (2, 4):
    m = mask_for("rec", rows=grp["hiding (1, 6, 7, 8)"], cols=[src])
    acc_b = np.mean([((D9 + (G9 @ (v * m).float()).double()) > 0).all(1).double().mean().item() for v in dws[:60]])
    print(f"    from neuron {src} alone (4 weights): {100 * (0.953 - acc_b):5.2f}")
