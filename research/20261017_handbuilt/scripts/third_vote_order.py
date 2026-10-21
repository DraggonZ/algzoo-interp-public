"""ARC: does the third's vote survive the second's later arrival? Vote of the third at the end =
centred score at its slot minus the same with the third replaced by -0.46 (a typical small
number), divided by (x3 + 0.46). Split by order and, for third-after-second, by the third's
ratio to the second at arrival (above or below the neuron-5 switch near 0.8)."""
import numpy as np
import torch as th

from algzoo.core import example, forward

P = example()
g = th.Generator().manual_seed(9)
x = th.randn(400000, 10, generator=g, dtype=th.float64)
o = x.argsort(-1)
p1, p2, p3 = o[:, -1], o[:, -2], o[:, -3]
r = th.arange(len(x))
x3 = x[r, p3]
keep = (x3 > 0) & (p3 < 9) & (p2 < 9)
x, p1, p2, p3, x3, r = x[keep], p1[keep], p2[keep], p3[keep], x3[keep], th.arange(int(keep.sum()))
xr = x.clone(); xr[r, p3] = -0.46
s, sr = forward(P, x), forward(P, xr)
c = lambda a: a - a.mean(-1, keepdim=True)
vote3 = ((c(s) - c(sr))[r, p3] / (x3 + 0.46)).numpy()
xr2 = x.clone(); xr2[r, p2] = -0.46
s2 = forward(P, xr2)
vote2 = ((c(s) - c(s2))[r, p2] / (x[r, p2] + 0.46)).numpy()
xn = x.numpy(); p2n, p3n, p1n = p2.numpy(), p3.numpy(), p1.numpy()
m2_at3 = np.array([np.sort(np.r_[0.0, 0.0, xn[i, :p3n[i]]])[-2] for i in range(len(xn))])
ratio = xn[np.arange(len(xn)), p3n] / np.maximum(m2_at3, 1e-9)
tie = (xn[np.arange(len(xn)), p2n] - xn[np.arange(len(xn)), p3n]) < 0.1
picks3 = (s.argmax(-1) == p3).numpy()
print("case                                         n      third's vote   second's vote   picks third (gap<0.1)")
for lab, m in (("third before second", p3n < p2n),
               ("third after second, above 0.8 of second", (p3n > p2n) & (ratio > 0.85)),
               ("third after second, below 0.7 of second", (p3n > p2n) & (ratio < 0.65))):
    print(f"  {lab:42s} {m.sum():6d}   {np.median(vote3[m]):6.1f}         {np.median(vote2[m]):6.1f}          {picks3[m & tie].mean():.3f}")
print("\nthird before second, by what the third was at arrival and whether the largest came between:")
was_sec = ratio >= 1  # third beat the second-so-far at its arrival (it became the running second or largest)
for lab, m in (("largest before both", (p1n < p3n)), ("largest between them", (p1n > p3n) & (p1n < p2n)), ("largest after both", p1n > p2n)):
    mm = (p3n < p2n) & m
    print(f"  {lab:22s} n={mm.sum():6d}  third's vote {np.median(vote3[mm]):6.1f}  second's vote {np.median(vote2[mm]):6.1f}  picks third (gap<0.1) {picks3[mm & tie].mean():.3f}")
