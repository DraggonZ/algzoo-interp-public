"""Review 3 check: the note's per-copy Spearman 0.81 between two-spike loss and n8 + n1 coefficient shift
(coeff_vs_loss.py) pools three noise levels. Both quantities grow with sigma, so recompute within each sigma."""
import sys

import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, "/data/output/research/algzoo/20261016_run2/scripts")
from coeff_compare import C, table  # noqa: E402
from algzoo.bench import load_bench  # noqa: E402
from algzoo.core import example  # noqa: E402

names, acc, models = load_bench(("mult",))
sig = np.array([float(str(n).split("_")[0][4:]) for n in names])
Tb, ab = table(example())
loss, shift = [], []
for m in models:
    Tc, ac = table(m)
    da = np.abs(Tc[:, :, 0] - Tb[:, :, 0]) / np.maximum(np.abs(Tb[:, :, 0]), 1.0)
    loss.append((ab - ac).mean())
    shift.append(da[[C.index(8), C.index(1)]].sum())
loss, shift = np.array(loss), np.array(shift)
print(f"all 96 copies: Spearman {spearmanr(loss, shift)[0]:+.2f}")
for s in (0.005, 0.01, 0.02):
    k = sig == s
    print(f"sigma {s}: Spearman {spearmanr(loss[k], shift[k])[0]:+.2f} (32 copies)")
np.savez("coeff_loss_within_sigma.npz", loss=loss, shift=shift, sig=sig)
