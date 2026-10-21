import sys
import time
import exact_nd as E
from algzoo.core import load
for name, ref in [("2nd_argmax_2_4_1073741824_bo5.pth", 0.32802305), ("2nd_argmax_3_4_1073741824_bo5.pth", 0.43328767), ("2nd_argmax_16_4_1073741824_bo5.pth", 0.99830735)]:
    params = load(name)
    for ang, q in [(0.3, 3), (0.2, 3), (0.3, 4)]:
        E.MAX_ANGLE, E.Q_PTS = ang, q
        t0 = time.time()
        acc, st = E.exact_accuracy(params)
        print(f"{name[:18]} angle {ang} q {q}: {acc:.8f} (MC {ref:.8f}) diff {acc - ref:+.1e} ops {st['ops']:.3g} ({time.time() - t0:.1f}s)", flush=True)
