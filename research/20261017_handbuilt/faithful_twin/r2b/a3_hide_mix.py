"""Analysis: the hiding neurons' reference (job 7's: -(recurrent input) / input weight), fitted on the largest so far M,
the second-largest so far S and a constant, steps 1-7 as the scorer. Prints the coefficients for ARC and for the
r1a-3 build at several comparator gains g, plus own accuracy (200,000 lists) and on-rates.
Question: how large is ARC's weight on the second-largest, and what does a twin with a visible wave cost?"""
import numpy as np
from algzoo.core import example
from algzoo.handbuilt_v25 import build_v25

rng = np.random.default_rng(7); X = rng.standard_normal((200_000, 10)); n = len(X)
M = np.full((n, 10), np.nan); S = np.full((n, 10), np.nan)
for t in range(1, 10):
    v = -np.sort(-X[:, :t], 1); M[:, t] = v[:, 0]
    if t >= 2: S[:, t] = v[:, 1]
rec = (X > np.nan_to_num(M, nan=-np.inf)) & (X > 0); rec[:, 0] = False


def report(name, W, hid):
    wi, wh, wo = W
    h = np.zeros((n, len(wi))); H = []
    for t in range(10):
        h = np.maximum(h @ wh.T + X[:, t:t + 1] * wi, 0); H.append(h)
    H = np.stack(H, 1)
    acc = ((H[:, -1] @ wo.T).argmax(1) == np.argsort(-X, 1)[:, 1]).mean()
    out = []
    for j in hid:
        ref = (-(H[:, 1:8] @ wh[j]) / wi[j]).reshape(-1)
        A = np.c_[M[:, 2:9].reshape(-1), S[:, 2:9].reshape(-1), np.ones(n * 7)]
        cf, *_ = np.linalg.lstsq(A, ref, rcond=None); r2 = 1 - np.var(A @ cf - ref) / np.var(ref)
        on = (H[:, 1:, j] > 0)[~rec[:, 1:]].mean(); off = 1 - (H[:, 1:, j] > 0)[rec[:, 1:]].mean()
        out.append(f"[{cf[0]:+.2f} M {cf[1]:+.2f} S, R2 {r2:.2f}, off@rec {off:.2f}, on {on:.2f}]")
    print(f"{name:12s} acc {acc:.4f}  " + " ".join(out))


report("ARC", [np.asarray(a, float) for a in example()], [1, 6, 7, 8])
for g in (1e4, 100.0, 30.0, 10.0, 3.0):
    net = build_v25(eps=0.1, theta=0.75 * 0.58, g=g) if False else None
    from algzoo.handbuilt_v23 import build_v23  # noqa
    import algzoo.handbuilt_v25 as v25
    # build_v25 does not pass g through; rebuild with v23's g and v25's two edits
    net = v25.build_v23(w=0.0, U_mem=1.0, share=0.25, rho=0.99, level_exact=False, g=g)
    wi, wh, wo = net.params
    Sx, D, G, L = (net.idx[k] for k in ("store", "detect", "keep", "linear"))
    a, s, beta, theta = 1000.0, 20.0, 0.25, 0.75 * 0.58
    wh[L, D] += 0.1 / a
    for M_, sg in ((G, -1.0), (L, 1.0)):
        wh[M_, Sx] += sg * beta * theta / s; wh[M_, D] += sg * beta * theta / a
    report(f"g={g:g}", [np.asarray(p, float) for p in net.params], [net.idx[k] for k in ("flip", "clock0", "clock1", "clock2")])
