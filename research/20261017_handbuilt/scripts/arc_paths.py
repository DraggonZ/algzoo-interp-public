"""Split ARC's final scores into contributions from one neuron's activity at one step, propagated exactly
through the list's own on/off pattern (no biases, so this split is exact). Lists: (a) a lone new largest 1.0 at
p (fillers -0.5); (b) a kept 1.0 at p after 2.0 at 0. Contributions of neuron 4 at step p, neuron 5 at step
p + 1, and the four cancellers at step p + 1, each centred over scores."""
import numpy as np
from algzoo.core import example

wi, wh, wo = (np.asarray(w, dtype=float) for w in example()); wi = wi.reshape(-1)
np.set_printoptions(precision=0, suppress=True, linewidth=160)


def contrib(x, j_list, tau):
    hs, masks = [], []
    h = np.zeros(16)
    for v in x:
        pre = wh @ h + wi * v; masks.append(pre > 0); h = np.maximum(pre, 0); hs.append(h)
    v = np.zeros(16); v[j_list] = hs[tau][j_list]
    for t in range(tau + 1, 10):
        v = masks[t] * (wh @ v)
    s = wo @ v
    return s - s.mean()


for p in (3, 5):
    a = np.full(10, -0.5); a[p] = 1.0
    b = np.full(10, -0.5); b[0] = 2.0; b[p] = 1.0
    print(f"--- position {p}")
    print("new largest: via neuron 4 at p   ", contrib(a, [4], p))
    print("new largest: via neuron 5 at p+1 ", contrib(a, [5], p + 1))
    print("new largest: via 4 and 5 together", contrib(a, [4], p) + contrib(a, [5], p + 1))
    print("new largest: cancellers at p+1   ", contrib(a, [1, 6, 7, 8], p + 1))
    print("kept number: via neuron 4 at p   ", contrib(b, [4], p))
    print("kept number: via neuron 5 at p+1 ", contrib(b, [5], p + 1))
    print("kept number: cancellers at p+1   ", contrib(b, [1, 6, 7, 8], p + 1))
