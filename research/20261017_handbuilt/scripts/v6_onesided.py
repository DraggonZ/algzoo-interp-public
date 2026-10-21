"""Relative votes plus a one-sided revealed wave (ARC's shape, `cancellers/n5_sharpens.log`): an old largest P at p
beaten at q shows A*P at p, A*P*s at positions p+1..8 (spill toward later positions), and -A*P*e at earlier
positions. Is one-sided spill harmless? Accuracy and agreement with ARC."""
import numpy as np, torch as th
from algzoo.core import example, forward

rng = np.random.default_rng(21)
X = rng.standard_normal((300_000, 10)); B, n = X.shape
o = X.argsort(1); tgt, third = o[:, -2], o[:, -3]
srt = np.sort(X, 1); gap = (srt[:, -2] - srt[:, -3]) / np.abs(srt[:, -2]).clip(1e-9)
adj = np.abs(tgt - third) == 1
arc = forward(example(), th.tensor(X)).numpy().argmax(1)
z = np.zeros((B, 2))
m2p = np.stack([np.sort(np.c_[z, X[:, :t]], 1)[:, -2] for t in range(n)], 1)
m1p = np.stack([np.max(np.c_[z, X[:, :t]], 1) for t in range(n)], 1)
record = X > m1p; final = X == X.max(1, keepdims=True)
idx = np.arange(10)


def run(c, A, s, e, symmetric=False):
    v = np.where(record, 0.0, np.maximum((1 - c) * X - c * np.maximum(m2p - X, 0), 0))
    v[:, 9] = np.where(record[:, 9], -1e9, v[:, 9])
    sc = v.copy()
    for p in range(9):
        P = np.where(record[:, p] & ~final[:, p], X[:, p], 0.0)
        if not P.any(): continue
        shape = np.where(idx == p, 1.0, np.where(idx > p, s, -e)); shape[9] = 0
        if symmetric: shape = np.where(idx == p, 1.0, s); shape[9] = 0
        sc += A * P[:, None] * shape[None, :]
    Pp = sc.argmax(1); near = gap < .1
    return (Pp == tgt).mean(), (Pp == arc).mean(), 1 - (Pp[adj & near] == tgt[adj & near]).mean(), 1 - (Pp[~adj & near] == tgt[~adj & near]).mean()


c = 0.75
for s in (0.0, 0.3, 0.6, 0.9):
    for e in (0.0, 1.0):
        r = max(((run(c, A, s, e), A) for A in (0.2, 0.25, 0.3, 0.4, 0.5)), key=lambda q: q[0][0])
        (acc, ag, ad, ap), A = r
        print(f"spill later {s:.1f}, push earlier down {e:.1f}, best A {A}: acc {acc:.4f}, agree ARC {ag:.3f}, near-tie adj {ad:.2f} apart {ap:.2f}")
r = max(((run(c, A, 0.6, 0, True), A) for A in (0.2, 0.25, 0.3, 0.4, 0.5)), key=lambda q: q[0][0])
print(f"for comparison, symmetric spill 0.6 everywhere: acc {r[0][0]:.4f}")
