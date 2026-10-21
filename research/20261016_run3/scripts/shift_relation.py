"""Weight-level test of the slide in every model: is output row j close to output row j-1 times one step matrix?

Step matrix = recurrent weights with the neurons off that are off on the most common quiet pattern after two inputs.
Reports the median cosine over j = 1..9 (centered output rows), and the same for a random rotation of the output rows
as a baseline (rows replaced by random vectors with the same norms).
"""
from collections import Counter

import numpy as np

from algzoo.core import example, zoo


def common_pattern(w_hi, w_hh):
    pats = Counter()
    for r in (0.3, 0.7):
        for p1 in range(10):
            for p2 in range(10):
                if p1 == p2:
                    continue
                x = np.zeros(10); x[p1] = 1.0; x[p2] = r
                h = np.zeros(len(w_hi))
                for t in range(10):
                    pre = w_hh @ h + x[t] * w_hi; h = np.maximum(0, pre)
                    if t > max(p1, p2):
                        pats[tuple((pre > 0).astype(int))] += 1
    return np.array(pats.most_common(1)[0][0], dtype=float)


rng = np.random.default_rng(0)
models = [("ARC example (16)", example())] + [(f"16-neuron seed {s}", zoo(16, 10, s)) for s in range(5)] + \
         [(f"8-neuron seed {s}", zoo(8, 10, s)) for s in range(3)] + [(f"32-neuron seed {s}", zoo(32, 10, s)) for s in range(3)]
for name, p in models:
    w_hi, w_hh, w_oh = (np.asarray(q, np.float64) for q in p)
    M = common_pattern(w_hi, w_hh)[:, None] * w_hh
    Wc = w_oh - w_oh.mean(0)
    cs = [np.dot(Wc[j], Wc[j - 1] @ M) / (np.linalg.norm(Wc[j]) * np.linalg.norm(Wc[j - 1] @ M)) for j in range(1, 10)]
    R = rng.standard_normal(Wc.shape); R -= R.mean(0)
    cr = [np.dot(R[j], R[j - 1] @ M) / (np.linalg.norm(R[j]) * np.linalg.norm(R[j - 1] @ M)) for j in range(1, 10)]
    print(f"{name:18s}: median cosine {np.median(cs):.2f} (min {np.min(cs):.2f}); random rows {np.median(cr):+.2f}")
