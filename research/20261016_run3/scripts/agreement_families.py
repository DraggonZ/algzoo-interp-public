"""Goal 2 (review 6 direction 5): does mean-field reduction keep accuracy while changing individual answers, in other model
families? Same reduction as meanfield.py (2nd-argmax: top three kept; argmedian: three ranks nearest the median kept,
the rest at their conditional means). Reports accuracy change, the share of answers that change, and mistakes caused and
fixed by the replaced inputs. Sampled, 2^22 inputs per family."""
import sys

import torch as th

sys.path.insert(0, "/data/output/research/algzoo/20261016_run3/scripts")
from meanfield import reduced, run  # noqa: E402

from algzoo.core import example, load, zoo  # noqa: E402

dev = "cuda"
N = 2 ** 22
g = th.Generator(device=dev).manual_seed(99)
fams = [("2nd_argmax 16x10 (ARC's model, seeds 0-4)", 10, "2nd", [example()] + [zoo(16, 10, s) for s in range(5)]),
        ("2nd_argmax 32x10", 10, "2nd", [load(f"2nd_argmax_32_10_1073741824_{s}.pth") for s in range(5)]),
        ("argmedian 32x11", 11, "med", [load(f"argmedian_32_11_1073741824_{s}.pth") for s in range(5)])]
for lab, n, task, ms in fams:
    X = th.randn(N, n, device=dev, generator=g)
    tgt = X.argsort(1)[:, -2] if task == "2nd" else X.argsort(1)[:, n // 2]
    lo, hi = (n - 3, n) if task == "2nd" else (n // 2 - 1, n // 2 + 2)
    Xr = reduced(X, lo, hi)
    for i, p in enumerate(ms):
        a, r = run(p, X), run(p, Xr)
        ra, rr = a == tgt, r == tgt
        print(f"{lab} #{i}: accuracy {100 * ra.float().mean():.2f}%, reduced minus full {100 * (rr.float().mean() - ra.float().mean()):+.2f} points; "
              f"answers changed {100 * (a != r).float().mean():.1f}%; replaced inputs cause {100 * (~ra & rr).float().mean():.2f} points of mistakes and fix {100 * (ra & ~rr).float().mean():.2f}", flush=True)
