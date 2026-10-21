"""Counts of numbers to copy from ARC, by combinations (eigen-directions) and by single weights, for several accuracy
budgets; checked in the real network with random rule errors (README experiment 18)."""
import numpy as np
import torch as th
from torch.func import jacrev, vmap
from algzoo.core import example

dev = "cuda"
W0 = [np.asarray(p, float) for p in example()]
w0 = th.tensor(np.concatenate([w.ravel() for w in W0]), device=dev)
names = [f"input->{j}" for j in range(16)] + [f"{i}->{j}" for j in range(16) for i in range(16)] + \
        [f"{i}->out{k}" for k in range(10) for i in range(16)]


def logits(w, x):
    wi, wh, wo = w[:16], w[16:272].view(16, 16), w[272:].view(10, 16)
    h = th.zeros(16, dtype=w.dtype, device=w.device)
    for t in range(10):
        h = th.relu(wh @ h + wi * x[t])
    return wo @ h


def batch_logits(w, X):
    wi, wh, wo = w[:16], w[16:272].view(16, 16), w[272:].view(10, 16)
    h = th.zeros(len(X), 16, dtype=w.dtype, device=w.device)
    for t in range(10):
        h = th.relu(h @ wh.T + X[:, t, None] * wi)
    return h @ wo.T


X = th.randn(200_000, 10, generator=th.Generator(device=dev).manual_seed(31), device=dev, dtype=th.float64)
F = th.zeros(432, 432, device=dev, dtype=th.float64)
jac = vmap(jacrev(logits), in_dims=(None, 0))
for chunk in X.split(5000):
    J = jac(w0, chunk) * w0
    p = th.softmax(batch_logits(w0, chunk), -1)
    Jm = J - (p[:, :, None] * J).sum(1, keepdim=True)
    F += th.einsum("nk,nka,nkb->ab", p, Jm, Jm)
F = (F / len(X)).cpu().numpy()
ev, V = np.linalg.eigh(F); ev, V = np.clip(ev[::-1], 0, None), V[:, ::-1]
diag = np.diag(F)

N = 2**22
Xa = th.randn(N, 10, generator=th.Generator(device=dev).manual_seed(7), device=dev)
Y = Xa.argsort(-1)[:, -2]
lb = th.log_softmax(batch_logits(w0.float(), Xa), -1)
acc0 = (lb.argmax(-1) == Y).float().mean().item()


@th.no_grad()
def score(e):  # e: relative change per weight (numpy)
    l = th.log_softmax(batch_logits((w0 * (1 + th.tensor(e, device=dev))).float(), Xa), -1)
    return 100 * (acc0 - (l.argmax(-1) == Y).float().mean().item()), (lb.exp() * (lb - l)).sum(-1).mean().item()


rng = np.random.default_rng(1)
cal = []
for sigma in [0.0005, 0.001, 0.0015, 0.002, 0.003, 0.004, 0.005, 0.0075, 0.01, 0.015]:
    for _ in range(6):
        cal.append(score(rng.standard_normal(432) * sigma))
cal = np.array(cal)  # (points lost, KL)
order = np.argsort(cal[:, 1])
# smooth monotone calibration: median points lost in KL bins, then interpolate
kl_sorted, pts_sorted = cal[order, 1], cal[order, 0]
bins = np.array_split(np.arange(len(kl_sorted)), 12)
kl_b = np.array([np.median(kl_sorted[b]) for b in bins]); pt_b = np.maximum.accumulate([np.median(pts_sorted[b]) for b in bins])
print("calibration, KL against accuracy loss (median per bin of 5 copies):")
print("  " + ", ".join(f"KL {k:.4f}: {p:.2f} pt" for k, p in zip(kl_b, pt_b)))
budgets = {}
for pt in [0.1, 0.25, 0.5, 1.0, 2.0]:
    budgets[pt] = float(np.interp(pt, pt_b, kl_b))
print("  KL budget for each accuracy budget: " + ", ".join(f"{p} pt -> {k:.4f}" for p, k in budgets.items()))

tail_ev = np.append(np.cumsum(ev[::-1])[::-1], 0)
dorder = np.argsort(-diag); tail_d = np.append(np.cumsum(diag[dorder][::-1])[::-1], 0)


def count(tail, r, B):
    return int(np.argmax(0.5 * r**2 * tail <= B))  # smallest k with 1/2 r^2 sum_{i>=k} <= B


rs = [0.005, 0.01, 0.02, 0.05, 0.1, 0.25, 0.5, 1.0]
print(f"\nsum of eigenvalues {ev.sum():.4e} = sum of diagonal {diag.sum():.4e}")
for lab, tail in [("combinations (eigen-directions)", tail_ev), ("single weights (by diagonal stiffness)", tail_d)]:
    print(f"\nnumbers to copy from ARC, {lab}; rows: accuracy budget, columns: rule error per weight")
    print("  budget  " + " ".join(f"{100 * r:>6.1f}%" for r in rs))
    for pt, B in budgets.items():
        print(f"  {pt:4.2f} pt " + " ".join(f"{count(tail, r, B):7d}" for r in rs))
print("\nstiffest single weights (share of the total):")
for i in dorder[:15]:
    print(f"  {names[i]:12s} {100 * diag[i] / diag.sum():5.2f}%")
groups = {"input": range(0, 16), "recurrent": range(16, 272), "readout": range(272, 432)}
k = count(tail_d, 0.05, budgets[0.5])
print(f"single weights to copy at 5% error, 0.5 point ({k}): " + ", ".join(f"{g} {sum(1 for i in dorder[:k] if i in idx)}" for g, idx in groups.items()))

print("\ncheck in the real network (budget 0.5 point): accuracy loss over 8 random rule errors, median (range)")
for r in [0.01, 0.05, 0.25]:
    kc, kw = count(tail_ev, r, budgets[0.5]), count(tail_d, r, budgets[0.5])
    res = {"no copying": [], f"copy {kc} combinations": [], f"copy {kw} single weights": []}
    for _ in range(8):
        e = rng.standard_normal(432) * r
        res["no copying"].append(score(e)[0])
        Vk = V[:, :kc]; res[f"copy {kc} combinations"].append(score(e - Vk @ (Vk.T @ e))[0])
        e2 = e.copy(); e2[dorder[:kw]] = 0; res[f"copy {kw} single weights"].append(score(e2)[0])
    for lab, v in res.items():
        print(f"  r = {100 * r:4.1f}%, {lab:28s}: {np.median(v):6.2f} points ({min(v):.2f} to {max(v):.2f})")
