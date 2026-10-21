"""ARC: lasting vote of a number at the end (replacement measure: replace it by -0.46, change in its
slot's centred score per unit) vs how many later arrivals got lasting votes (beat 0.8 of the
second-so-far, or new largest). Only numbers that themselves got a lasting vote, are not the final
largest, are not the last input; grouped by their age at the end too."""
import numpy as np
import torch as th

from algzoo.core import example, forward

P = example()
g = th.Generator().manual_seed(14)
x = th.randn(60000, 10, generator=g, dtype=th.float64)
xn = x.numpy()
B, n = xn.shape
z = np.zeros((B, 2))
m2p = np.stack([np.sort(np.c_[z, xn[:, :t]], 1)[:, -2] for t in range(n)], 1)
m1p = np.stack([np.max(np.c_[z, xn[:, :t]], 1) for t in range(n)], 1)
lasting = (xn > 0.8 * m2p) & (xn > 0)
amax = xn.argmax(1)
c = lambda a: a - a.mean(-1, keepdim=True)
s = c(forward(P, x))
rows = []
for p in range(9):
    xr = x.clone(); xr[:, p] = -0.46
    sr = c(forward(P, xr))
    vote = ((s - sr)[:, p] / (x[:, p] + 0.46)).numpy()
    later = lasting[:, p + 1:9].sum(1)  # later lasting arrivals, not counting the last input
    ok = lasting[:, p] & (amax != p) & (xn[:, p] > 0.2)
    later_beats = (xn[:, p + 1:] > xn[:, p:p + 1]).sum(1)  # how many later numbers are bigger
    rows.append(np.c_[np.full(ok.sum(), 9 - p), later[ok], later_beats[ok], vote[ok]])
r = np.concatenate(rows)
print("later lasting votes   n      median vote per unit")
for k in range(5):
    m = r[:, 1] == k
    print(f"      {k}             {m.sum():6d}     {np.median(r[m, 3]):6.1f}")
print("\nlater numbers bigger than it (0 = it is the largest... excluded; 1 = it ends second)")
for k in range(1, 5):
    m = r[:, 2] == k
    print(f"      {k}             {m.sum():6d}     {np.median(r[m, 3]):6.1f}")
print("\nby later lasting votes, only those that end as the second (one bigger number later or earlier... rank 2):")
