"""Stiff directions of the readout: which combinations of the 160 readout weights decide near-tie lists?

Hidden states do not depend on the readout, so each list's score leads are exactly linear in it. On 200,000 lists, for
every (list, rival) pair with |lead of the correct position over the rival| < 2, the lead's gradient is +h on the
correct row and -h on the rival's row. Covariance of these gradients -> eigenvectors (stiff directions first).
Then correct the readout rule of readout_shift.py (readout_rule.json) along the top k: rule - P_k (rule - ARC), and
measure the paired accuracy cost on 2^22 lists for each k. Control: the same number k of random directions."""
import json
import numpy as np
import torch as th
from algzoo.core import example, forward

wi, wh, wo = (np.asarray(p, float) for p in example())
G = np.array(json.load(open("readout_rule.json"))["rule"])
x = th.randn(200000, 10, dtype=th.float64, generator=th.Generator().manual_seed(5))
_, H = forward((wi, wh, wo), x, return_hidden=True)
h = H[:, -1].numpy(); y = x.argsort(-1)[:, -2].numpy()
S = h @ wo.T
C = np.zeros((160, 160)); npairs = 0
for rv in range(10):
    lead = S[np.arange(len(y)), y] - S[:, rv]
    sel = (np.abs(lead) < 2) & (y != rv)
    g = np.zeros((sel.sum(), 10, 16))
    g[np.arange(sel.sum()), y[sel]] = h[sel]
    g[:, rv] -= h[sel]
    g = g.reshape(len(g), -1); C += g.T @ g; npairs += len(g)
ev, V = np.linalg.eigh(C); ev, V = ev[::-1], V[:, ::-1]
print(f"{npairs} near-tie (list, rival) pairs; share of gradient variance in the top k directions: " +
      ", ".join(f"k={k}: {ev[:k].sum() / ev.sum():.3f}" for k in [1, 2, 5, 10, 20, 40, 80]))

dev = "cuda"; N = 2**22
X = th.randn(N, 10, generator=th.Generator(device=dev).manual_seed(7), device=dev)
Y = X.argsort(-1)[:, -2]
with th.no_grad():
    a, b = (th.as_tensor(p, dtype=th.float32, device=dev) for p in (wi, wh))
    hh = th.zeros(N, 16, device=dev)
    for t in range(10):
        hh = th.relu(hh @ b.T + X[:, t, None] * a)
acc = lambda W: ((hh @ th.as_tensor(W, dtype=th.float32, device=dev).T).argmax(-1) == Y).float().mean().item()
acc0 = acc(wo)
E = (G - wo).reshape(-1)
rng = np.random.default_rng(0)
print(f"ARC {acc0:.5f}; rule alone costs {100 * (acc0 - acc(G)):.2f} points")
for k in [0, 5, 10, 20, 30, 40, 60, 80, 100, 120]:
    Vk = V[:, :k]
    Ek = E - Vk @ (Vk.T @ E)
    Q = np.linalg.qr(rng.standard_normal((160, max(k, 1))))[0][:, :k]
    Er = E - Q @ (Q.T @ E)
    print(f"  correct top {k:3d} stiff directions: cost {100 * (acc0 - acc(wo + Ek.reshape(10, 16))):6.2f} points "
          f"(remaining error {np.linalg.norm(Ek) / np.linalg.norm(E):.2f} of the rule's); {k} random directions: "
          f"{100 * (acc0 - acc(wo + Er.reshape(10, 16))):6.2f}")
