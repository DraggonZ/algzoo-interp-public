"""Where do v23's memory slots switch off with the imprint at one vote (U = 1)? For every memory pre-activation below
zero: the step, the label the slot holds, that label relative to the current largest's position, and whether the label
holds a record, the current largest, a kept positive or a negative number."""
import numpy as np
from collections import Counter
from algzoo.handbuilt_v23 import build_v23
from realise_check import pre_acts
import sys; sys.path.insert(0, "../v23")
if __name__ == "__main__":
    net = build_v23(U_mem=1.0, share=0.25); Yi = [net.idx[f"mem{j}"] for j in range(8)]
    X = np.random.default_rng(9).standard_normal((50000, 10)); Z = pre_acts(net, X)
    c, k = Counter(), Counter(); tot = 0
    for t in range(10):
        r = X[:, :t + 1].argmax(1)                                 # current largest's position at step t
        for j, yi in enumerate(Yi):
            q = t - 2 - j
            if q < 0: continue
            m = Z[:, t, yi] < -1e-9; tot += m.sum()
            if not m.any(): continue
            rel = q - r[m]; c.update(rel.tolist())
            kind = np.where(q == r[m], "current largest", np.where(X[m, q] >= np.maximum.accumulate(X, 1)[m, q], "earlier record", np.where(X[m, q] > 0, "positive", "negative")))
            k.update(kind.tolist())
    print(f"memory neuron-steps switched off (slots in use): {tot:,} of {50000 * 8 * 10:,}")
    print("by label minus the current largest's position:", sorted(c.items()))
    print("by what the label holds:", k.most_common())
