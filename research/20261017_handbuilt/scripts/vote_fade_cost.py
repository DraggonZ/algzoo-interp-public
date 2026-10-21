"""How much does a fading code cost, with the best fixed discount? Votes fade by rho per step of age (age = steps
from arrival to the end). Kept non-record x votes rho^age * (x - c * second-so-far)^+; an old largest P votes
rho^age * P when beaten; the current largest does not vote. Best c per rho, chosen on one half, scored on the other."""
import numpy as np

rng = np.random.default_rng(8)
x = rng.standard_normal((1_000_000, 10)); B, n = x.shape
z = np.zeros((B, 2))
m2p = np.stack([np.sort(np.c_[z, x[:, :t]], 1)[:, -2] for t in range(n)], 1)
m1p = np.stack([np.max(np.c_[z, x[:, :t]], 1) for t in range(n)], 1)
record = x > m1p; final = x == x.max(1, keepdims=True)
tgt = x.argsort(1)[:, -2]; age = n - 1 - np.arange(n)
A, Bh = slice(0, B // 2), slice(B // 2, B)
for rho in (1.0, 0.95, 0.9, 0.85, 0.8, 0.7):
    f = rho ** age
    best = None
    for c in np.round(np.arange(0, 1.0, 0.05), 2):
        v = np.where(record, np.where(final, 0.0, x), (x - c * m2p).clip(0)) * f
        a = (v[A].argmax(1) == tgt[A]).mean()
        if best is None or a > best[0]: best = (a, c, (v[Bh].argmax(1) == tgt[Bh]).mean())
    v0 = np.where(record, np.where(final, 0.0, x), x.clip(0)) * f
    print(f"fade {rho:4.2f} per step: no discount {(v0[Bh].argmax(1) == tgt[Bh]).mean():.4f}; best discount c = {best[1]:.2f}: {best[2]:.4f}")
