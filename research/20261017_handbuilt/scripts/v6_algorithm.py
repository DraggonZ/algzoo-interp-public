"""Trained-network layout as an algorithm, no sharp new-largest switch, no dip.
Kept non-records vote k_rel = (1 - c) x - c (m2 - x)^+ on their own position (forget if <= 0); the true second
always gets (1 - c) s. Old largests, when beaten, show a wave A * P * shape(distance) where shape is a cosine on
the 9-cycle (what a 3-neuron clock can hide), optionally sharpened. Current largest hidden. Position 9: the last
input's relative vote. Compared with ARC."""
import numpy as np, torch as th
from algzoo.core import example, forward

rng = np.random.default_rng(19)
X = rng.standard_normal((300_000, 10)); B, n = X.shape
o = X.argsort(1); tgt, third = o[:, -2], o[:, -3]
srt = np.sort(X, 1); gap = (srt[:, -2] - srt[:, -3]) / np.abs(srt[:, -2]).clip(1e-9)
adj = np.abs(tgt - third) == 1
arc = forward(example(), th.tensor(X)).numpy().argmax(1)
z = np.zeros((B, 2))
m2p = np.stack([np.sort(np.c_[z, X[:, :t]], 1)[:, -2] for t in range(n)], 1)
m1p = np.stack([np.max(np.c_[z, X[:, :t]], 1) for t in range(n)], 1)
record = X > m1p; final = X == X.max(1, keepdims=True)
D = np.arange(9)[:, None] - np.arange(9)[None, :]


def run(c, A, shape):
    v = np.where(record, 0.0, np.maximum((1 - c) * X - c * np.maximum(m2p - X, 0), 0))
    s = np.zeros((B, 10)); s[:, :9] = v[:, :9]
    s[:, 9] = np.where(record[:, 9], -1e9, v[:, 9])
    K = shape[np.abs(D) % 9]                      # 9x9 spread of a wave
    rev = np.where(record & ~final, X, 0.0)[:, :9]
    s[:, :9] += A * rev @ K
    P = s.argmax(1)
    near = gap < .1
    return (P == tgt).mean(), (P == arc).mean(), 1 - (P[adj & near] == tgt[adj & near]).mean(), 1 - (P[~adj & near] == tgt[~adj & near]).mean()


ang = np.deg2rad(40.0) * np.arange(9)
shapes = {"spike (ideal)": np.eye(9)[0], "cosine (3-neuron clock)": np.cos(ang),
          "cosine + 2nd harmonic": (np.cos(ang) + np.cos(2 * ang)) / 2}
print("ARC: acc 0.953; near-tie error adjacent 0.15, apart 0.15")
for c in (0.5, 0.75):
    for name, sh in shapes.items():
        best = max(((run(c, A, sh), A) for A in (0.1, 0.25, 0.5, 0.75, 1, 1.5, 2, 3)), key=lambda r: r[0][0])
        (acc, agree, ad, ap), A = best
        print(f"c={c} {name:24s} best A={A:<4}: acc {acc:.4f}, agree with ARC {agree:.3f}, near-tie error adjacent {ad:.2f}, apart {ap:.2f}")
