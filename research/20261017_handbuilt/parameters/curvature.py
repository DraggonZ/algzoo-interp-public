"""Curvature count (README experiment 17): Fisher matrix of ARC's output in relative weight coordinates, its spectrum,
free directions, and counts of directions that matter at rule precisions r. 200,000 lists (float64)."""
import numpy as np
import torch as th
from torch.func import jacrev, vmap
from algzoo.core import example

dev = "cuda"
W0 = [np.asarray(p, float) for p in example()]
w0 = th.tensor(np.concatenate([w.ravel() for w in W0]), device=dev)
names = [f"input->{j}" for j in range(16)] + [f"{i}->{j}" for j in range(16) for i in range(16)] + \
        [f"{i}->out{k}" for k in range(10) for i in range(16)]


def logits(w, x):  # one list
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


gen = th.Generator(device=dev).manual_seed(31)
X = th.randn(200_000, 10, generator=gen, device=dev, dtype=th.float64)
F = th.zeros(432, 432, device=dev, dtype=th.float64)
jac = vmap(jacrev(logits), in_dims=(None, 0))
for chunk in X.split(5000):
    J = jac(w0, chunk) * w0          # relative coordinates: d logits / d log|w|
    p = th.softmax(batch_logits(w0, chunk), -1)
    Jm = J - (p[:, :, None] * J).sum(1, keepdim=True)   # J - p^T J  (centred over outputs)
    F += th.einsum("nk,nka,nkb->ab", p, Jm, Jm)
F /= len(X)
ev, V = th.linalg.eigh(F)
ev, V = ev.flip(0).cpu().numpy(), V.flip(1).cpu().numpy()
print(f"largest eigenvalue {ev[0]:.3e}; eigenvalues by rank: " + ", ".join(f"{k}: {ev[k - 1]:.2e}" for k in [1, 2, 5, 10, 20, 50, 100, 200, 300, 368, 390, 400, 401, 410, 420, 432]))
rel = ev / ev[0]
for thr in [1e-6, 1e-8, 1e-10, 1e-12]:
    print(f"  eigenvalues below {thr:.0e} of the largest: {(rel < thr).sum()}")
gaps = np.log10(np.clip(rel[:-1], 1e-30, None)) - np.log10(np.clip(rel[1:], 1e-30, None))
k = np.argmax(gaps[300:]) + 300
print(f"  biggest drop in the spectrum after rank 300: between rank {k + 1} ({rel[k]:.1e}) and {k + 2} ({rel[k + 1]:.1e})")

# calibrate KL -> accuracy with random relative noise of all weights
N = 2**22
Xa = th.randn(N, 10, generator=th.Generator(device=dev).manual_seed(7), device=dev)
Y = Xa.argsort(-1)[:, -2]


@th.no_grad()
def score(w):
    lb = th.log_softmax(batch_logits(w0.float(), Xa), -1)
    l = th.log_softmax(batch_logits(w.float(), Xa), -1)
    return (l.argmax(-1) == Y).float().mean().item(), (lb.exp() * (lb - l)).sum(-1).mean().item()


acc0, _ = score(w0)
rng = np.random.default_rng(0)
pts = []
for sigma in [0.001, 0.002, 0.003, 0.005, 0.0075]:
    for rep in range(4):
        e = th.tensor(rng.standard_normal(432), device=dev) * sigma
        a, kl = score(w0 * (1 + e))
        pts.append((kl, acc0 - a))
pts = np.array(pts)
slope = (pts[:, 0] @ pts[:, 1]) / (pts[:, 0] @ pts[:, 0])
KL05 = 0.005 / slope
print(f"\ncalibration (random relative noise, 20 copies): accuracy loss ~ {slope:.3f} x KL; KL for 0.5 point = {KL05:.4f}")
for kl, d in pts[::4]:
    print(f"  KL {kl:.4f}: loss {100 * d:.2f} points (predicted {100 * slope * kl:.2f})")
# check the quadratic prediction on the same copies: KL vs 1/2 e^T F e
e = th.tensor(rng.standard_normal(432), device=dev) * 0.003
print(f"  quadratic check, one copy at 0.3%: measured KL {score(w0 * (1 + e))[1]:.4f}, 1/2 e F e {0.5 * (e @ F @ e).item():.4f}")

print("\ncounts of directions that matter at rule precision r (relative error per unit direction):")
for r in [0.005, 0.01, 0.02, 0.05, 0.1, 0.25]:
    thr = 2 * KL05 / r**2
    print(f"  r = {100 * r:4.1f}%: N(r) = {(ev > thr).sum():3d}; effective dimension {np.sum(ev / (ev + thr)):6.1f}")

groups = {"input": range(0, 16), "recurrent": range(16, 272), "readout": range(272, 432)}
print("\nwhere the top directions load (share of squared loading by kind of weight), and their top weights:")
for i in [0, 1, 2, 3, 4, 9, 19, 49]:
    sh = {g: (V[list(idx), i] ** 2).sum() for g, idx in groups.items()}
    top = np.argsort(-np.abs(V[:, i]))[:5]
    print(f"  direction {i + 1:3d} (eigenvalue {ev[i]:.2e}): " + ", ".join(f"{g} {s:.2f}" for g, s in sh.items()) +
          " | " + ", ".join(f"{names[t]} {V[t, i]:+.2f}" for t in top))
np.save("curvature_eigenvalues.npy", ev)
open("curvature_eigenvalues.txt", "w").write("\n".join(f"{x:.6e}" for x in ev))
