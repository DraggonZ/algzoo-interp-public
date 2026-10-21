"""Readout rule: row k = s_k * r @ M^k + u, with M = D W_hh for ARC's most common on/off pattern at steps 4-8.

Fits (to ARC's readout weights, least squares; no accuracy used):
  anchored: r = ARC's centred row 0, u = ARC's row mean, s_k fitted (9 numbers);
  global:   r (16), s_k (9), u (16, a free shift) fitted by alternating least squares.
Cost: paired accuracy change on 2^22 lists with the rule's readout in place of ARC's. Also per-group relative error of
the fitted readout (columns from store/detector, gates, hiding, memory)."""
from collections import Counter
import numpy as np
import torch as th
from algzoo.core import example

W0 = [np.asarray(p, float) for p in example()]
wi, wh, wo = W0
Xs = np.random.default_rng(17).standard_normal((100_000, 10))
h = np.zeros((len(Xs), 16)); masks = []
for t in range(10):
    h = np.maximum(h @ wh.T + Xs[:, t:t + 1] * wi, 0); masks.append(h > 0)
top, cnt = Counter(map(lambda r: r.tobytes(), np.concatenate(masks[4:9]))).most_common(1)[0]
on = np.frombuffer(top, dtype=bool)
print(f"most common on/off pattern at steps 4-8 ({100 * cnt / (5 * len(Xs)):.1f}% of steps): off neurons {list(np.where(~on)[0])}")
M = np.diag(on.astype(float)) @ wh
Mk = [np.linalg.matrix_power(M, k) for k in range(10)]

dev = "cuda"
N = 2**22
X = th.randn(N, 10, generator=th.Generator(device=dev).manual_seed(7), device=dev)
Y = X.argsort(-1)[:, -2]


@th.no_grad()
def acc(wo_new):
    a, b, c = (th.as_tensor(p, dtype=th.float32, device=dev) for p in (wi, wh, wo_new))
    hh = th.zeros(N, 16, device=dev)
    for t in range(10):
        hh = th.relu(hh @ b.T + X[:, t, None] * a)
    return ((hh @ c.T).argmax(-1) == Y).float().mean().item()


acc0 = acc(wo)
groups = {"store/detector": [2, 4], "gates": [5, 9], "hiding": [1, 6, 7, 8], "memory": [0, 3, 10, 11, 12, 13, 14, 15]}


def report(name, R, nfree):
    Rc, Wc = R - R.mean(0), wo - wo.mean(0)
    cosk = [Rc[k] @ Wc[k] / np.linalg.norm(Rc[k]) / np.linalg.norm(Wc[k]) for k in range(10)]
    print(f"\n{name} ({nfree} fitted numbers): cost {100 * (acc0 - acc(R)):.2f} points; "
          f"cosine with ARC's centred rows by position {np.round(cosk, 2)}")
    print("  relative error (centred, by source group): " + ", ".join(
        f"{g} {np.linalg.norm((Rc - Wc)[:, ns]) / np.linalg.norm(Wc[:, ns]):.2f}" for g, ns in groups.items()))


def fit_s(r, target):
    return np.array([(target[k] @ (r @ Mk[k])) / ((r @ Mk[k]) @ (r @ Mk[k])) for k in range(10)])


print(f"ARC {acc0:.5f}")
u0 = wo.mean(0); Wc = wo - u0
# anchored at row 0
r = Wc[0]; s = fit_s(r, Wc); s[0] = 1
report("anchored at position 0", np.stack([s[k] * r @ Mk[k] for k in range(10)]) + u0, 9)

# global alternating fit of r, s, u
r, s, u = Wc[0].copy(), np.ones(10), u0.copy()
for it in range(500):
    T = wo - u
    A = np.concatenate([s[k] * Mk[k].T for k in range(10)])  # (160, 16): rows k of T = s_k M_k^T r
    r = np.linalg.lstsq(A, T.reshape(-1), rcond=None)[0]
    s = fit_s(r, T); s /= s[0]; r *= 1  # fix the scale of r by s_0 = 1 next round
    pred = np.stack([s[k] * r @ Mk[k] for k in range(10)])
    u = (wo - pred).mean(0)
G = pred + u
print(f"\nglobal fit residual (relative, uncentred): {np.linalg.norm(G - wo) / np.linalg.norm(wo):.3f}; scales s_k {np.round(s, 2)}")
report("global fit", G, 16 + 9 + 16)

# how precise must the readout be? ARC's readout plus a fraction of the global fit's error
for f in [0.02, 0.05, 0.1, 0.25, 0.5]:
    print(f"  ARC + {f:.2f} x (rule - ARC): cost {100 * (acc0 - acc(wo + f * (G - wo))):.2f} points")

import json
json.dump({"rule": G.tolist(), "r": r.tolist(), "s": s.tolist(), "u": u.tolist()}, open("readout_rule.json", "w"))
