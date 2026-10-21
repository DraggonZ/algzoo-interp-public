"""Monte Carlo reference accuracies (2^32 samples each, float32 forward, fixed seeds)."""
import sys

import numpy as np

from algzoo.core import accuracy, load

N = 2**32
for i, name in enumerate(sys.argv[1:]):
    acc, n = accuracy(load(name), N, batch=2**24, seed=7000 + i)
    print(f"{name}: MC {acc:.8f} ± {np.sqrt(acc * (1 - acc) / n):.2e} (n = {n})", flush=True)
