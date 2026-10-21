"""ARC neuron 3 on two-spike inputs (largest 1 at p1, second r at p2, others at background):
its value at each step, and which pairs fail when it is held at 0 from step k on."""
import itertools

import numpy as np
import torch as th

from algzoo.core import example, forward

P = [th.tensor(p) for p in example()]
w_hi, w_hh, w_oh = P


def run(x, kill=False):
    h = th.zeros(len(x), 16, dtype=x.dtype); hs = []
    for t in range(10):
        h = th.relu(h @ w_hh.T + x[:, t, None] * w_hi)
        if kill:
            h[:, 3] = 0
        hs.append(h.clone())
    return h @ w_oh.T, th.stack(hs, 1)


for bg in (-0.46,):
    for rr in (0.5, 0.9):
        pairs = list(itertools.permutations(range(10), 2))
        x = th.full((90, 10), bg, dtype=th.float64)
        for k, (a, b) in enumerate(pairs):
            x[k, a], x[k, b] = 1.0, rr
        s, H = run(x)
        sk, _ = run(x, kill=True)
        ok, okk = s.argmax(-1).numpy(), sk.argmax(-1).numpy()
        tgt = np.array([b for _, b in pairs])
        print(f"\nbackground {bg}, ratio {rr}: correct normal {np.sum(ok == tgt)}/90, neuron 3 killed {np.sum(okk == tgt)}/90")
        F = np.zeros((10, 10), dtype=object)
        T = np.full((10, 10), "  .")
        for k, (a, b) in enumerate(pairs):
            T[a, b] = " ok" if okk[k] == tgt[k] else f"{okk[k]:3d}"
        print("with neuron 3 killed, rows = largest's slot, cols = second's slot (number = slot chosen when wrong)")
        print("\n".join("".join(row) for row in T))
        n3_end = np.full((10, 10), np.nan)
        for k, (a, b) in enumerate(pairs):
            n3_end[a, b] = H[k, -1, 3].item()
        print("neuron 3 at the end (rows largest, cols second):")
        print(np.array2string(n3_end, precision=1, max_line_width=200))
