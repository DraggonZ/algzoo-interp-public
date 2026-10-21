"""Methods study 2 (calibration): the additive vote model explained ARC's successes but only 11-28% of its
mistakes. What does it do on the hand-built network, whose mechanism is fully known?
Same features as additive_vote_fit.py (type x age, fitted to the network's choices by softmax regression),
with the lasting-vote cut at 0.8 (as for ARC) and at 1.0 (the hand-built network's true cut)."""
import numpy as np
import torch as th

from algzoo import handbuilt as hb
from algzoo.core import example, forward

g = th.Generator().manual_seed(16)
x = th.randn(400_000, 10, generator=g, dtype=th.float64)
xn = x.numpy(); B, n = xn.shape
z = np.zeros((B, 2))
m2p = np.stack([np.sort(np.c_[z, xn[:, :t]], 1)[:, -2] for t in range(n)], 1)
m1p = np.stack([np.max(np.c_[z, xn[:, :t]], 1) for t in range(n)], 1)
amax = xn.argmax(1); tgt = xn.argsort(1)[:, -2]
later_bigger = np.stack([(xn[:, t + 1:] > xn[:, t:t + 1]).sum(1) for t in range(n)], 1)


def features(cut, shrink):
    typ = np.where(((xn > cut * m2p) & (xn > 0)) | (xn > m1p), 1, 2)
    typ[np.arange(B), amax] = 0
    typ[:, -1] = np.where(amax == n - 1, 0, 3)
    if shrink:
        typ = np.where(typ == 1, 1 + 4 * np.minimum(later_bigger, 3), typ)
    T = typ.max() + 1
    F = np.zeros((B, T * n))
    for t in range(n):
        F[np.arange(B), typ[:, t] * n + (n - 1 - t)] += xn[:, t]
    return F


def fit_and_score(choice, F):
    Ft = th.tensor(F, device="cuda", dtype=th.float32); y = th.tensor(choice, device="cuda")
    W = th.zeros(F.shape[1], n, device="cuda", requires_grad=True); opt = th.optim.Adam([W], lr=0.05)
    for _ in range(3000):
        idx = th.randint(0, 300_000, (65536,), device="cuda")
        loss = th.nn.functional.cross_entropy(Ft[idx] @ W, y[idx]); opt.zero_grad(); loss.backward(); opt.step()
    with th.no_grad():
        P = (Ft[300_000:] @ W).argmax(1).cpu().numpy()
    c, t = choice[300_000:], tgt[300_000:]
    wrong = c != t
    return (P == c).mean(), (P == t).mean(), (P[wrong] == c[wrong]).mean(), wrong.mean()


for name, Pm in (("hand-built (gains 100)", hb.build_v2(100.0, 100.0).params), ("ARC", example())):
    choice = forward(Pm, x).argmax(-1).numpy()
    for cut in (0.8, 1.0):
        for shrink in (False, True):
            agree, acc, repro, errrate = fit_and_score(choice, features(cut, shrink))
            print(f"{name:22s} cut {cut} {'with shrink' if shrink else 'no shrink  '}: agrees with the network {agree:.3f}; accuracy {acc:.3f}; "
                  f"network's mistakes reproduced {repro:.2f} (network error rate {errrate:.3f})")
