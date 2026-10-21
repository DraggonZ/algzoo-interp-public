"""True accuracies of the unseen check set: 16-neuron, 10-input models trained on 2^20 ... 2^29 sequences, seeds 0-4.
2^26 random inputs each (standard error at most about 0.006 points). Computed before any estimate is run on them."""
import numpy as np

from algzoo.core import accuracy, zoo

rows = []
for e in range(20, 30):
    for s in range(5):
        p = zoo(16, 10, s, n_seqs=2 ** e)
        a, n = accuracy(p, 2 ** 26, batch=2 ** 22, seed=500 + 10 * e + s)
        rows.append((e, s, a))
        print(f"n_seqs 2^{e} seed {s}: accuracy {a:.5f}", flush=True)
R = np.array(rows)
np.save("/data/output/research/algzoo/20261016_run3/reports/fresh_truth.npy", R)
