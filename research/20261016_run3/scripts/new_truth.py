"""Step 1 of the new-seed check: true accuracies of the 16 newly trained models (2^26 inputs each), saved before scoring."""
import glob

import numpy as np

from algzoo.core import accuracy, load

files = sorted(glob.glob("/data/output/research/algzoo/20261016_run3/models/2nd_argmax_16_10_1073741824_*.pth"))
rows = []
for f in files:
    seed = int(f.rsplit("_", 1)[1].split(".")[0])
    a, n = accuracy(load(f), n_samples=2 ** 26, seed=seed)
    rows.append((seed, a)); print(f"seed {seed}: true accuracy {a:.4f} ({n} inputs)", flush=True)
np.save("/data/output/research/algzoo/20261016_run3/reports/new_truth.npy", np.array(rows))
