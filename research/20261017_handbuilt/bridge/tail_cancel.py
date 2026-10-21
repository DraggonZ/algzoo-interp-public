"""Question: after a record at r is overtaken at q, the trained memory keeps the old record's wave at every position,
including positions after q (record_profile.log). Do the cancelling neurons, restarted at q, cancel that tail?
Lists: record at r (beats the largest before it by 0.05 and is above 0.3), overtaken at q by a number that stays the
largest; nudge the record at r; per position, final-score change through the memory and through the cancelling
neurons, median over lists."""
import numpy as np
from algzoo.core import example
wi, wh, wo = example()
rng = np.random.default_rng(3); X = rng.standard_normal((1500000, 10))
def final(X):
    h = np.zeros((len(X), 16))
    for t in range(10): h = np.maximum(h @ wh.T + X[:, t:t + 1] * wi, 0)
    return h
MEM = [0, 3, 10, 11, 12, 13, 14, 15]; CAN = [1, 6, 7, 8]
eps = 1e-3
for r, q in [(1, 4), (2, 5), (3, 5), (2, 7), (4, 7)]:
    pre = np.maximum(X[:, :r].max(1), 0)
    ok = (X[:, r] > pre + 0.05) & (X[:, r] > 0.3) & (X[:, r + 1:q].max(1) < X[:, r]) & (X[:, q] > X[:, r] + 0.05)
    if q < 9: ok &= X[:, q + 1:].max(1) < X[:, q]
    Xs = X[ok]; Xp = Xs.copy(); Xp[:, r] += eps
    dh = (final(Xp) - final(Xs)) / eps
    mem = np.median(dh[:, MEM] @ wo[:, MEM].T, 0); can = np.median(dh[:, CAN] @ wo[:, CAN].T, 0); tot = np.median(dh @ wo.T, 0)
    print(f"record at {r}, overtaken at {q} (n={ok.sum()}); positions 0-9")
    for nm, v in (("memory", mem), ("cancel", can), ("total", tot)):
        print(f"   {nm:7s}", " ".join(f"{x:7.1f}" for x in v))
    tail = np.arange(10) > q
    print(f"   after the overtaking (positions {q+1}-9): memory sum {mem[tail].sum():6.1f}, cancellers sum {can[tail].sum():6.1f}, total {tot[tail].sum():6.1f}")
