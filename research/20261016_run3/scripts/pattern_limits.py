"""Limit (i): estimators that handle on/off patterns one at a time. Review 5's check C on all six trained models (2^23 inputs).

For each model: the number of distinct patterns (which neurons are on at each of the 10 steps) that hold sampled mistakes;
the Good-Turing estimate of the mistake probability in patterns never seen (patterns with one sampled mistake / inputs);
and, for budgets B of runs, the mistake probability outside the B patterns holding the most mistakes, against sampling's
error at budget B. A pattern-by-pattern method that spends at least one run per pattern cannot recover the mistakes outside
the patterns it handles, so if that leftover exceeds sampling's error at every B, the class loses to sampling.
"""
import sys

import numpy as np
import torch as th

from algzoo.core import example, zoo

dev = "cuda"
N, CH = 2 ** 23, 2 ** 21
SH = th.arange(64, device=dev, dtype=th.int64)


def keys(bits):
    """Pack [n, 160] bools into [n, 3] int64 exactly."""
    b = bits.to(th.int64)
    parts = [b[:, i:i + 64] for i in (0, 64, 128)]
    return th.stack([(p << SH[: p.shape[1]]).sum(1) for p in parts], 1)


for name in ["example", 0, 1, 2, 3, 4]:
    p = example() if name == "example" else zoo(16, 10, name)
    w_hi, w_hh, w_oh = (th.as_tensor(q, dtype=th.float32, device=dev) for q in p)
    g = th.Generator(device=dev).manual_seed(123)
    wk, ak = [], []
    nw = 0
    for _ in range(N // CH):
        X = th.randn(CH, 10, device=dev, generator=g)
        h = th.zeros(CH, 16, device=dev); bits = []
        for t in range(10):
            h = th.relu(h @ w_hh.T + X[:, t, None] * w_hi); bits.append(h > 0)
        k = keys(th.cat(bits, 1))
        wrong = (h @ w_oh.T).argmax(1) != X.argsort(1)[:, -2]
        wk.append(k[wrong].cpu()); ak.append(k.cpu()); nw += int(wrong.sum())
    uw, cw = th.unique(th.cat(wk), dim=0, return_counts=True)
    ua = th.unique(th.cat(ak), dim=0)
    c = np.sort(cw.numpy())[::-1]
    err = nw / N
    n1 = int((c == 1).sum())
    gt = n1 / N
    cs = np.cumsum(c)
    line = []
    for B in [100, 1000, 10000, 100000]:
        left = (nw - (cs[B - 1] if B <= len(c) else nw)) / N
        samp = np.sqrt(err * (1 - err) / B)
        line.append(f"B={B}: left {100 * left:.2f} vs sampling {100 * samp:.2f}")
    print(f"{name}: error {100 * err:.2f} pts; patterns seen {len(ua)}, with mistakes {len(uw)}; half the mistakes need "
          f"{np.searchsorted(cs / nw, 0.5) + 1}; Good-Turing unseen mistake probability {100 * gt:.2f} pts "
          f"({gt / err:.0%} of the error) | " + "; ".join(line))
    sys.stdout.flush()
