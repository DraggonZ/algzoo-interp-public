"""Share of lists of each kind (common.kinds plus the two kinds added in partial_by_step.py), on the scoring lists
(seed 4) and the partial-switching lists (seed 2626), 200,000 each."""
import numpy as np
from common import lists, kinds
for seed in (4, 2626):
    X = lists(seed, 200_000); K = kinds(X); o = np.argsort(-X, 1)
    K["answer right after the largest"] = o[:, 1] == o[:, 0] + 1; K["answer two or more after the largest"] = o[:, 1] > o[:, 0] + 1
    print(f"seed {seed}: " + " | ".join(f"{k.split(' (')[0]} {m.mean():.3f}" for k, m in K.items()))
