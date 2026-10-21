"""How many directions in weight space decide a noisy copy's accuracy? Same setup as hilton_noise.py (ARC's noise:
every weight plus N(0, 0.0025^2); 200 copies; 50,000 shared lists). The per-copy first-order estimate moves each list's
nine leads by G dw. Keep only the top r directions of G (eigenvectors of sum over lists and leads of g g^T, weighted
to the lists near a tie) and predict each copy from the r numbers u = V_r^T dw. Report per-copy RMSE and correlation
against the copies' true accuracy for r = 1, 2, 5, 10, 20, 50, 432, and which weights the top directions load on."""
import numpy as np
import torch as th
from torch.func import vmap, grad
from algzoo.core import example
dev = "cuda"; SIG = 0.0025
g = th.Generator(device=dev).manual_seed(43); N = 50000
X = th.randn(N, 10, device=dev, generator=g, dtype=th.float64); y = X.argsort(-1)[:, -2]
W = [th.as_tensor(np.asarray(w, float), dtype=th.float64, device=dev) for w in example()]
sizes = [w.numel() for w in W]; P = sum(sizes)
def batch_acc(Ws):
    wi, wh, wo = Ws; h = th.zeros(N, 16, dtype=th.float64, device=dev)
    for t in range(10): h = th.relu(h @ wh.T + X[:, t:t + 1] * wi)
    return ((h @ wo.T).argmax(1) == y).double().mean().item()
def split(v):
    out, i = [], 0
    for w in W: out.append(w + v[i:i + w.numel()].reshape(w.shape)); i += w.numel()
    return out
dws = [th.randn(P, device=dev, dtype=th.float64, generator=g) * SIG for _ in range(200)]
true = np.array([batch_acc(split(v)) for v in dws])
def logits(params, x):
    wi, wh, wo = params; h = th.zeros(16, dtype=x.dtype, device=x.device)
    for t in range(10): h = th.relu(wh @ h + wi * x[t])
    return wo @ h
jac = vmap(grad(lambda p, x, e: (logits(p, x) * e).sum()), in_dims=(None, 0, 0))
with th.no_grad(): L = vmap(logits, in_dims=(None, 0))(W, X)
oy = th.nn.functional.one_hot(y, 10).double(); keep = th.arange(10, device=dev).unsqueeze(0) != y.unsqueeze(1)
D9 = ((L * oy).sum(1, keepdim=True) - L)[keep].reshape(-1, 9)
G = th.zeros(N, 10, P, device=dev, dtype=th.float32)
for b in range(0, N, 5000):
    sl = slice(b, b + 5000)
    for k in range(10):
        G[sl, k] = th.cat([gi.reshape(5000, -1) for gi in jac(W, X[sl], oy[sl] - th.nn.functional.one_hot(th.full((5000,), k, device=dev), 10).double())], 1).float()
G9 = G[keep].reshape(N, 9, P); del G
size = SIG * G9.norm(dim=2)                                      # leads' noise size
near = (D9.abs().float() < 4 * size)                             # leads that can flip
Gn = G9[near]                                                    # [n_near_leads, P]
M = (Gn.double().T @ Gn.double()); ev, V = th.linalg.eigh(M); V = V[:, th.argsort(ev, descending=True)]; ev = th.sort(ev, descending=True).values
print(f"share of near-tie leads' noise variance in the top r directions: " + ", ".join(f"r={r}: {(ev[:r].sum() / ev.sum()).item():.3f}" for r in (1, 2, 5, 10, 20, 50)))
full = np.array([((D9 + (G9 @ v.float()).double()) > 0).all(1).double().mean().item() for v in dws])
print(f"all 432 directions: RMSE {100 * np.sqrt(np.mean((full - true) ** 2)):.2f} points, correlation {np.corrcoef(full, true)[0, 1]:.3f}; constant guess {100 * true.std():.2f}")
for r in (1, 2, 5, 10, 20, 50):
    Vr = V[:, :r].float(); GV = G9 @ Vr                          # [N, 9, r]
    est = np.array([((D9 + (GV @ (Vr.T @ v.float())).double()) > 0).all(1).double().mean().item() for v in dws])
    print(f"top {r:2d} directions: RMSE {100 * np.sqrt(np.mean((est - true) ** 2)):.2f} points, correlation {np.corrcoef(est, true)[0, 1]:.3f}")
names = []
for i in range(16): names.append(f"input->{i}")
for i in range(16):
    for j in range(16): names.append(f"{j}->{i}")
for p in range(10):
    for j in range(16): names.append(f"{j}->out{p}")
for d in range(3):
    v = V[:, d].cpu().numpy(); top = np.argsort(-np.abs(v))[:8]
    print(f"direction {d + 1}: " + ", ".join(f"{names[t]} {v[t]:+.2f}" for t in top))
# Interpretation test: four 'level shift' directions, one per hiding neuron: its incoming recurrent weights moved in
# proportion to each source neuron's mean activity (over steps 1-9 and lists), so the direction shifts that neuron's
# input by a near-constant amount. Predict each copy from the projection of its noise onto these four directions.
with th.no_grad():
    hs = []; h = th.zeros(N, 16, dtype=th.float64, device=dev)
    for t in range(10): h = th.relu(h @ W[1].T + X[:, t:t + 1] * W[0]); hs.append(h)
    mh = th.stack(hs[:-1], 1).mean((0, 1))                      # mean activity of each neuron as a source (steps 0-8)
U = []
for k in (1, 6, 7, 8):
    u = th.zeros(P, device=dev, dtype=th.float64); M_ = th.zeros(16, 16, device=dev, dtype=th.float64); M_[k] = mh
    u[16:16 + 256] = M_.flatten(); U.append(u / u.norm())
for kset, lab in (([1, 6, 7, 8], "the four hiding neurons"), ([7], "neuron 7 only"), ([1, 6, 7, 8, 0, 3, 10, 11, 12, 13, 14, 15], "hiding and memory neurons")):
    Us = []
    for k in kset:
        u = th.zeros(P, device=dev, dtype=th.float64); M_ = th.zeros(16, 16, device=dev, dtype=th.float64); M_[k] = mh
        u[16:16 + 256] = M_.flatten(); Us.append(u)
    Q, _ = th.linalg.qr(th.stack(Us, 1)); Qf = Q.float(); GQ = G9 @ Qf
    est = np.array([((D9 + (GQ @ (Qf.T @ v.float())).double()) > 0).all(1).double().mean().item() for v in dws])
    print(f"level shifts of {lab} ({len(kset)} numbers per copy): RMSE {100 * np.sqrt(np.mean((est - true) ** 2)):.2f} points, correlation {np.corrcoef(est, true)[0, 1]:.3f}")
cos = abs((V[:, 0] @ U[2]).item()); print(f"top eigen-direction against neuron 7's level-shift direction: |cosine| {cos:.2f}")
