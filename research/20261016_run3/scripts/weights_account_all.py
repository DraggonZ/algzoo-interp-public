"""Does the weight-level structure of ARC's model recur? For each of the 22 trained models: input weights above 1 in size,
split by sign; the readout of a unit drop on the negative-input neurons (newest slot minus the others' mean); and the
tiny-model agreement at zero background from new_mechanism.log where available."""
import glob

import numpy as np

from algzoo.core import example, load, zoo

models = [("example", example())] + [(f"seed{s}", zoo(16, 10, s)) for s in range(5)]
models += [(f.rsplit("_", 1)[1].split(".")[0], load(f)) for f in sorted(glob.glob("/data/output/research/algzoo/20261016_run3/models/2nd_argmax_16_10_1073741824_*.pth"))]
for name, p in models:
    w_hi, w_hh, w_oh = (np.asarray(q, np.float64) for q in p)
    Wc = w_oh - w_oh.mean(0)
    neg = np.flatnonzero(w_hi < -1); pos = np.flatnonzero(w_hi > 1)
    v = Wc[:, neg] @ w_hi[neg] * -1          # change in scores per unit input when the negative-input neurons drop
    print(f"{name}: negative input weights {len(neg)} ({', '.join(f'{w_hi[k]:.1f}' for k in neg)}); positive {len(pos)} ({', '.join(f'{w_hi[k]:.1f}' for k in pos)}); "
          f"vote from the weights: newest slot minus others {-(v[9] - np.delete(v, 9).mean()):+.0f}, top slot {(-v).argmax()}", flush=True)
