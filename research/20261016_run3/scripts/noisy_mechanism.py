"""Why noisy copies fail, in terms of the mechanism.

For each of the 96 copies: vote placed right at the later input's arrival, then held or lost while sliding (90 pairs x 3
ratios), lone-input score spread, and how far the ring's rotation speeds are from even spacing. Compared with the exact
two-spike accuracy and the true accuracy from run 2.
"""
import sys

import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, "/data/output/research/algzoo/20261016_run3/scripts")
from mechanism_checks import checks  # noqa: E402

from algzoo.bench import load_bench  # noqa: E402

Z = np.load("/data/output/research/algzoo/20261016_run2/reports/z2_check.npz", allow_pickle=True)
z2 = dict(zip([str(n) for n in Z["names"]], Z["z2_exact"]))
names, acc, models = load_bench(("mult",))


def ring_misfit(speeds):
    """Fit nonzero speeds to whole multiples of one base speed; return the base and the worst miss in cycles/step."""
    s = np.array([v for v in speeds if v > 0.02])
    best = (9, 0)
    for f0 in np.linspace(0.08, 0.16, 801):
        k = np.maximum(1, np.round(s / f0))
        miss = np.abs(s - k * f0).max()
        if miss < best[0]:
            best = (miss, f0)
    return best


rows = []
for nm, a, p in zip(names, acc, models):
    o = checks(str(nm), p)
    placed, held, n = o["placed_held"]
    miss, f0 = ring_misfit(o["ring"])
    rows.append((str(nm), a, z2[str(nm)], placed / n, held / n, (placed - held) / n, o["slide"], o["lone_vs_two_spread"][0], miss, f0))
    print(f"{nm:12s} acc {a:.3f} Z2 {z2[str(nm)]:.3f} | placed {placed/n:.2f} held {held/n:.2f} lost while sliding {(placed-held)/n:.2f} "
          f"slide {o['slide']:.2f} lone spread {o['lone_vs_two_spread'][0]:5.1f} | ring base {f0:.3f} worst miss {miss:.3f}", flush=True)
R = np.array([r[1:] for r in rows], dtype=float)
lab = ["accuracy", "Z2", "placed", "held", "lost sliding", "slide", "lone spread", "ring miss", "ring base"]
for j in range(2, 9):
    print(f"Spearman with Z2: {lab[j]:12s} {spearmanr(R[:, 1], R[:, j])[0]:+.2f}   with accuracy {spearmanr(R[:, 0], R[:, j])[0]:+.2f}")
np.save("/data/output/research/algzoo/20261016_run3/reports/noisy_mechanism.npy", R)
