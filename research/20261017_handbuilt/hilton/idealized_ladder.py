"""Hilton's estimate, first rungs. Idealize blocks of the trained weights by the account's rules, then predict the
trained network's accuracy from the idealized network with the weight gaps dW = W_trained - W_ideal:
  (A) first-order correction: each lead D_k becomes D_k + sum_j (dD_k/dw_j) dW_j (exact gaps, linearized);
  (B) gaps as noise: each gap treated as a random amount of the same size, so the nine leads are jointly normal with
      covariance sum_j (dD_k/dw_j)(dD_l/dw_j) dW_j^2; predicted accuracy = mean P(all nine > 0) (as noise_joint.py).
Sensitivities come from the idealized network's own backward passes on 20,000 random lists. Truth: the trained
network's accuracy on the same lists.
Rungs (each adds to the previous):
  1. store and detector by rule: block [[1, s/a], [-a/s, -1]], s = 17 (the store holds about 17 x the largest),
     a = 10.16 (the detector's input weight); their weights from all other neurons set to 0 (with argument "keep":
     left as trained, so only the 2x2 block and the two input weights are idealized; gaps at most 0.11).
  2. hiding block (1, 6, 7, 8 among themselves) reduced to its one rotation: the leading complex eigenpair kept
     (about 43 degrees per step, 0.94), the other two directions set to 0."""
import numpy as np
import torch as th
from torch.func import vmap, grad
from algzoo.core import example
import sys
KEEP_EXTRAS = len(sys.argv) > 1 and sys.argv[1] == "keep"
dev = "cuda"; g = th.Generator(device=dev).manual_seed(21); N = 20000
X = th.randn(N, 10, device=dev, generator=g, dtype=th.float64); y = X.argsort(-1)[:, -2]
wi0, wh0, wo0 = (np.asarray(p, float) for p in example())
def rung(k):
    wi, wh, wo = wi0.copy(), wh0.copy(), wo0.copy()
    if k >= 1:
        s, a = 17.0, 10.16
        if not KEEP_EXTRAS:
            for r in (2, 4): wh[r, :] = 0.0
        wh[2, 2], wh[2, 4], wh[4, 2], wh[4, 4] = 1.0, s / a, -a / s, -1.0
        wi[2], wi[4] = 0.0, a
    if k >= 2:
        H = [1, 6, 7, 8]; B = wh[np.ix_(H, H)]; ev, V = np.linalg.eig(B); i = np.argmax(np.abs(ev))
        lead = [i, int(np.argmin(np.abs(ev - np.conj(ev[i]))))]
        Vi = np.linalg.inv(V); B2 = sum(ev[j] * np.outer(V[:, j], Vi[j]) for j in lead).real
        wh[np.ix_(H, H)] = B2
        print(f"   hiding block: kept eigenvalues {np.round(ev[lead], 3)}, dropped {np.round(np.delete(ev, lead), 3)}")
    return wi, wh, wo
def logits(params, x):
    wi, wh, wo = params; h = th.zeros(16, dtype=x.dtype, device=x.device)
    for t in range(10): h = th.relu(wh @ h + wi * x[t])
    return wo @ h
jac = vmap(grad(lambda p, x, e: (logits(p, x) * e).sum()), in_dims=(None, 0, 0))
oy = th.nn.functional.one_hot(y, 10).double(); keep = th.arange(10, device=dev).unsqueeze(0) != y.unsqueeze(1)
T = [th.as_tensor(w, dtype=th.float64, device=dev) for w in (wi0, wh0, wo0)]
with th.no_grad(): truth = (vmap(logits, in_dims=(None, 0))(T, X).argmax(1) == y).double().mean().item()
print(f"trained network on these lists: {truth:.4f}")
for k in (1, 2):
    print(f"rung {k}:")
    Wi = [th.as_tensor(w, dtype=th.float64, device=dev) for w in rung(k)]; dW = [t - w for t, w in zip(T, Wi)]
    with th.no_grad(): L = vmap(logits, in_dims=(None, 0))(Wi, X)
    D = ((L * oy).sum(1, keepdim=True) - L)[keep].reshape(-1, 9)
    lin = th.zeros(N, 10, device=dev, dtype=th.float64); C = th.zeros(N, 10, 10, device=dev, dtype=th.float64)
    for b in range(0, N, 5000):
        sl = slice(b, b + 5000); Gs = []; Ls = []
        for kk in range(10):
            gr = jac(Wi, X[sl], oy[sl] - th.nn.functional.one_hot(th.full((5000,), kk, device=dev), 10).double())
            Gs.append(th.cat([(gi * d.unsqueeze(0)).reshape(5000, -1) for gi, d in zip(gr, dW)], 1))
        G = th.stack(Gs, 1); lin[sl] = G.sum(2); C[sl] = G @ G.transpose(1, 2)
    lin9 = lin[keep].reshape(-1, 9); C9 = C[keep.unsqueeze(2) & keep.unsqueeze(1)].reshape(-1, 9, 9)
    ideal = (D > 0).all(1).double().mean().item(); A = ((D + lin9) > 0).all(1).double().mean().item()
    Lc = th.linalg.cholesky(C9 + 1e-9 * th.eye(9, device=dev, dtype=th.float64)); pr = []
    for b in range(0, N, 5000):
        Z = th.randn(5000, 2000, 9, device=dev, dtype=th.float64, generator=g)
        pr.append(((D[b:b + 5000].unsqueeze(1) + Z @ Lc[b:b + 5000].transpose(1, 2)) > 0).all(2).double().mean(1))
    print(f"   idealized network's own accuracy {ideal:.4f}; (A) first-order correction {A:.4f}; (B) gaps as noise {th.cat(pr).mean().item():.4f}; truth {truth:.4f}")
