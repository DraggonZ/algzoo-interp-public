"""The sensitive weight directions of ARC (as hilton_lowrank.py: eigenvectors of the summed products of near-tie leads'
weight gradients) and, for the weights into each hiding neuron, where the sensitivity comes from: the gradient of a
lead with respect to W_hh[n, j] is sum_t delta_n(t) h_j(t-1), so it splits exactly by step t and by the steps since the
largest arrived (K). Saves dirs.npz (top directions, eigenvalues, the near-tie Gram matrix) for later scripts."""
import numpy as np, torch as th
from torch.func import vmap, grad
from noiselib import net, DEV, DT

W = net("ARC"); P = 432; SIG = 0.0025
g = th.Generator(device=DEV).manual_seed(43); N = 50000
X = th.randn(N, 10, device=DEV, dtype=DT, generator=g); y = X.argsort(-1)[:, -2]

def logits(params, x):
    wi, wh, wo = params; h = th.zeros(16, dtype=x.dtype, device=x.device)
    for t in range(10): h = th.relu(wh @ h + wi * x[t])
    return wo @ h
jac = vmap(grad(lambda p, x, e: (logits(p, x) * e).sum()), in_dims=(None, 0, 0))
with th.no_grad(): L = vmap(logits, in_dims=(None, 0))(W, X)
oy = th.nn.functional.one_hot(y, 10).double(); keep = th.arange(10, device=DEV).unsqueeze(0) != y.unsqueeze(1)
D9 = ((L * oy).sum(1, keepdim=True) - L)[keep].reshape(-1, 9)
G = th.zeros(N, 10, P, device=DEV, dtype=th.float32)
for b in range(0, N, 5000):
    sl = slice(b, b + 5000)
    for k in range(10):
        G[sl, k] = th.cat([gi.reshape(5000, -1) for gi in jac(W, X[sl], oy[sl] - th.nn.functional.one_hot(th.full((5000,), k, device=DEV), 10).double())], 1).float()
G9 = G[keep].reshape(N, 9, P); del G
near = (D9.abs().float() < 4 * SIG * G9.norm(dim=2))
Gn = G9[near].double(); M = Gn.T @ Gn
ev, V = th.linalg.eigh(M); o = th.argsort(ev, descending=True); ev, V = ev[o], V[:, o]
np.savez("dirs.npz", V=V[:, :50].cpu().numpy(), ev=ev.cpu().numpy(), M=M.cpu().numpy())
print("near-tie Gram matrix saved; top eigenvalue shares:", np.round((ev[:10] / ev.sum()).cpu().numpy(), 3))

# split of direction 1's row-7 part by step and K: contribution of (list, t) to sum over near-tie leads of g g^T projected
# on direction 1 is (g . v1) * dg/dW[7,:](t) . v1[row 7]; we instead report, for row n, the near-tie-weighted average of
# delta_n(t) h(t-1) by K, as a share of the row's total sensitivity along v1.
v1 = V[:, 0]
def run_states(X):
    wi, wh, wo = W; h = th.zeros(len(X), 16, device=DEV, dtype=DT); Hp, PRE = [], []
    for t in range(10):
        pre = h @ wh.T + X[:, t:t + 1] * wi; Hp.append(h); PRE.append(pre); h = th.relu(pre)
    return th.stack(Hp, 1), th.stack(PRE, 1)
Hp, PRE = run_states(X)
# per list: weight = sum over near leads of (g . v1)
w_list = ((G9.double() @ v1) * near.double()).sum(1)                          # [N]
# delta_n(t): gradient of that weighted lead combination w.r.t. pre_n(t); get it with autograd on pre-activations
def lead_combo_grad(X, coef):
    wi, wh, wo = W; h = th.zeros(len(X), 16, device=DEV, dtype=DT); pres = []
    for t in range(10):
        pre = (h @ wh.T + X[:, t:t + 1] * wi).requires_grad_(True); pres.append(pre); h = th.relu(pre)
    s = h @ wo.T
    tot = (coef * s).sum(); gr = th.autograd.grad(tot, pres)
    return th.stack(gr, 1)                                                     # [N, 10, 16]
# coefficient on each score: sum over near leads of (g.v1) * (e_y - e_k)
gv = (G9.double() @ v1) * near.double()                                        # [N, 9]
others = th.arange(10, device=DEV)[None].expand(N, 10)[keep].reshape(N, 9)
coef = th.zeros(N, 10, device=DEV, dtype=DT)
coef.scatter_add_(1, others, -gv); coef[th.arange(N), y] += gv.sum(1)
delta = lead_combo_grad(X, coef)                                               # [N, 10, 16]
pm = th.cummax(th.cat([th.full((N, 1), -1e9, device=DEV, dtype=DT), X], 1), 1)
mx = pm.values[:, 1:]; arg = pm.indices[:, 1:] - 1                              # largest up to t and its position
prevmax = pm.values[:, :-1]
Kt = th.arange(10, device=DEV)[None] - arg                                     # steps since the largest (0 = record now)
for n in (7, 6, 8, 1):
    rowv = v1[16 + 16 * n:16 + 16 * (n + 1)]
    contrib = (delta[:, :, n:n + 1] * Hp).sum(0) @ rowv                         # by step t: [10]
    tot = contrib.sum().item()
    byK = [((delta[:, :, n:n + 1] * Hp) @ rowv)[Kt == k].sum().item() for k in range(10)]
    print(f"row into neuron {n}: share of direction-1 sensitivity by step t: " + " ".join(f"{c / tot:+.2f}" for c in contrib.tolist()))
    print(f"    by steps since the largest K (0 = the step that sets a new largest): " + " ".join(f"{c / tot:+.2f}" for c in byK))
    # by whether the arriving number is within 10% of the largest before it (near switching of a comparator)
    close = ((X - prevmax).abs() < 0.1 * prevmax.abs()) & (prevmax > 0)
    print(f"    share at steps where the number is within 10% of the previous largest: {((delta[:, :, n:n + 1] * Hp) @ rowv)[close].sum().item() / tot:+.2f} (such steps: {close.double().mean().item():.3f})")
