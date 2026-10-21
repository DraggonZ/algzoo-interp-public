"""Budgeted exact enumeration on the other trained length-4 models (zoo seeds 0-4 of M16,4), against 2^31-sample truth."""
import numpy as np

import exact_nd as E
from algzoo.core import accuracy, zoo

for s in range(5):
    p = zoo(16, 4, s)
    truth, n = accuracy(p, 2**31, batch=2**24, seed=9500 + s)
    se = np.sqrt(truth * (1 - truth) / n)
    full, stf = E.exact_accuracy(p)
    line = f"seed {s}: truth {truth:.6f} ± {se:.1e} | exact {full:.6f} (z {(full - truth) / se:+.1f}, {stf['ops'] / 1152:.0f} samples)"
    for eps in (3e-6, 1e-5):
        a, st = E.exact_accuracy(p, prune=eps)
        B = st["ops"] / 1152
        samp = np.sqrt(truth * (1 - truth) / B)
        line += f" | prune {eps:.0e}: error {st['corrected'] - truth:+.1e} at {B:.0f} samples (sampling RMSE {samp:.1e})"
    print(line, flush=True)
