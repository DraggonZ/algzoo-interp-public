"""Review 3 check: M5,4 had z = -2.83 in table_exact_len4.log (float32 Monte Carlo); the claims table cites an unlogged
float64 re-check. Current exact code against 2^31-sample float64 Monte Carlo."""
import numpy as np
import torch as th

from algzoo.core import accuracy, load
from algzoo.exact_nd import exact_accuracy

name = "2nd_argmax_5_4_1073741824_bo5.pth"
p = load(name)
ex, st = exact_accuracy(p)
a, n = accuracy(p, n_samples=2**31, batch=2**24, seed=54, dtype=th.float64)
se = np.sqrt(a * (1 - a) / n)
print(f"{name}: exact {ex:.8f}; float64 MC {a:.8f} ± {se:.1e} (n = {n}); z {(ex - a) / se:+.2f}")
