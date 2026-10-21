"""Review 3 check: uncounted adjacency work in budgeted mode (prune threshold from argv) on M16,4."""
import sys
import time

import cost_audit as C  # noqa: F401  (runs the full exact audit first, installs the wrappers)
from algzoo.core import load

for thr in [float(a) for a in sys.argv[1:]] or [3e-6, 1e-5]:
    for k in C.extra:
        C.extra[k] = 0
    params = load("2nd_argmax_16_4_1073741824_bo5.pth")
    t0 = time.time()
    acc, st = C.E.exact_accuracy(params, prune=thr)
    print(f"prune {thr:g}: corrected {st['corrected']:.8f} uncorrected {acc:.8f}; counted {st['ops'] / C.per_sample:,.0f} samples; "
          f"uncounted adjacency {C.extra['adj'] / C.per_sample:,.0f} samples; qhull calls {C.dl['calls']:,} ({time.time() - t0:.1f}s)")
