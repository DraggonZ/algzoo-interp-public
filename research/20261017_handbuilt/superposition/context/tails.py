"""Where do the steep pieces come from? The smaller numbers' lead coefficients are heavy-tailed (99th percentile 21,
max 939, median 0.31). For each neuron, make it linear (activation = pre-activation, no switching) and recompute the
coefficients' 99th percentile on near ties; the neuron whose switching makes the tail should remove it. Also report
the network's accuracy with that neuron linear (cost of removing its switching)."""
import numpy as np
from flips import *
Xs = X[:100000]; Ss, Ts = S[:100000], T[:100000]; nn = torch.arange(len(Xs), device=DEV); sm = small[:100000]
def scores_lin(W, X, lin):
    h = torch.zeros(len(X), 16, device=DEV, dtype=dt)
    for t in range(10):
        z = h @ W[1].T + X[:, t:t + 1] * W[0]
        mask = torch.zeros(16, device=DEV, dtype=torch.bool); mask[lin] = True
        h = torch.where(mask, z, torch.relu(z))
    return h @ W[2].T
def jac_lin(lin):
    Y = Xs.clone().requires_grad_(True); Sc = scores_lin(ARC, Y, lin); lead = Sc[nn, Ss] - Sc[nn, Ts]
    c, = torch.autograd.grad(lead.sum(), Y); return c[sm].abs()
R = lists(1_000_000, seed=5); rs = R.argsort(1, descending=True)[:, 1]
base = jac_lin([]); print(f"no change: |c| 90/99/99.9% {np.round(torch.quantile(base, torch.tensor([0.9, 0.99, 0.999], device=DEV, dtype=dt)).cpu().numpy(), 1)}")
for k in range(16):
    q = torch.quantile(jac_lin([k]), torch.tensor([0.9, 0.99, 0.999], device=DEV, dtype=dt)).cpu().numpy()
    acc = (scores_lin(ARC, R, [k]).argmax(1) == rs).double().mean().item()
    print(f"neuron {k:2d} linear: |c| 90/99/99.9% {np.round(q, 1)} | accuracy {100*acc:.1f}%", flush=True)
