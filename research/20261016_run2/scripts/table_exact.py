"""Exact accuracy against 2^32-sample Monte Carlo for a list of zoo models (2nd argmax or argmedian)."""
import sys
import time

import numpy as np
import torch as th

from algzoo.core import forward, load
import exact_nd as E

N, B = 2**32, 2**24
for i, name in enumerate(sys.argv[1:]):
    params = load(name)
    n = params[2].shape[0]
    rank = (n + 1) // 2 if "argmedian" in name else 2
    t0 = time.time()
    ex, st = E.exact_accuracy(params, rank=rank)
    te = time.time() - t0
    g = th.Generator(device="cuda").manual_seed(8000 + i)
    hits = 0
    with th.no_grad():
        for _ in range(N // B):
            x = th.randn(B, n, generator=g, device="cuda", dtype=th.float64)
            hits += (forward(params, x).argmax(-1) == x.argsort(-1)[:, -rank]).sum().item()
    mc = hits / N
    se = np.sqrt(mc * (1 - mc) / N)
    d = params[1].shape[0]
    fwd = n * (d * d + 2 * d)
    print(f"{name}: exact {ex:.8f} MC {mc:.8f} ± {se:.1e} z {(ex - mc) / se:+.2f} | regions {st['regions']} ops {st['ops']:.3g}"
          f" = {st['ops'] / fwd:.3g} samples ({te:.0f}s)", flush=True)
