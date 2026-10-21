"""ARC's model: at which value does an arriving number switch off the lowering
neurons (1, 6, 7, 8)? Bin arrivals by x relative to the second-largest so far (m2)
and the largest so far (m1); report the mean share of the four switched off, and
how lasting the input's effect is (|d h_{t+1} / d x_t|)."""
import numpy as np

from algzoo.core import example

w_hi, w_hh, w_oh = example()
rng = np.random.default_rng(4)
x = rng.standard_normal((100000, 10))
B, n = x.shape
H = np.zeros((B, n + 1, 16))
for t in range(n):
    H[:, t + 1] = np.maximum(H[:, t] @ w_hh.T + x[:, t, None] * w_hi, 0)
L = [1, 6, 7, 8]
rows = []
for t in range(2, 9):
    pre = np.sort(x[:, :t], 1)
    m1, m2 = pre[:, -1], pre[:, -2]
    ok = (m2 > 0.2) & (x[:, t] > 0)
    off = (H[:, t + 1][:, L] == 0).mean(1)
    d1 = (H[:, t + 1] > 0) * w_hi
    d2 = (H[:, t + 2] > 0) * (d1 @ w_hh.T)
    rows.append(np.c_[x[ok, t] / m2[ok], x[ok, t] / m1[ok], off[ok], np.linalg.norm(d2[ok], axis=1)])
r = np.concatenate(rows)
print("x / second-so-far   share of 4 lowering neurons off   lasting effect |dh(t+1)/dx(t)|   (only x below largest)")
below_max = r[:, 1] < 1
for lo, hi in [(0, .5), (.5, .8), (.8, .9), (.9, .97), (.97, 1.0), (1.0, 1.03), (1.03, 1.1), (1.1, 1.3), (1.3, 2), (2, 9)]:
    m = below_max & (r[:, 0] >= lo) & (r[:, 0] < hi)
    print(f"  {lo:4.2f}-{hi:4.2f}  n={m.sum():6d}   off {r[m, 2].mean():.2f}   lasting {np.median(r[m, 3]):5.1f}")
print("new largest (x above largest so far):")
m = r[:, 1] >= 1
print(f"  n={m.sum()} off {r[m, 2].mean():.2f}  lasting {np.median(r[m, 3]):.1f}")

# which neurons change on/off status around x = 0.8 m2, at the arrival step and one step later?
print("\nactive share per neuron at arrival step (rows: x / second-so-far bins, x below largest)")
acts0, acts1, ratio = [], [], []
for t in range(2, 9):
    pre = np.sort(x[:, :t], 1)
    m1, m2 = pre[:, -1], pre[:, -2]
    ok = (m2 > 0.2) & (x[:, t] > 0) & (x[:, t] < m1)
    acts0.append(H[ok, t + 1] > 0); acts1.append(H[ok, t + 2] > 0); ratio.append(x[ok, t] / m2[ok])
acts0, acts1, ratio = map(np.concatenate, (acts0, acts1, ratio))
print("bin        " + " ".join(f"{k:4d}" for k in range(16)))
for lo, hi in [(0, .5), (.5, .7), (.7, .8), (.8, .9), (.9, 1.0), (1.0, 1.2), (1.2, 9)]:
    m = (ratio >= lo) & (ratio < hi)
    print(f"{lo:3.1f}-{hi:3.1f} t  " + " ".join(f"{v:4.2f}" for v in acts0[m].mean(0)))
