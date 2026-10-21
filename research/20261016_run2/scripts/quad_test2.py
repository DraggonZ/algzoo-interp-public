import time
import exact_nd as E
from algzoo.core import load
for name, ref in [("2nd_argmax_2_4_1073741824_bo5.pth", 0.3280274), ("2nd_argmax_3_4_1073741824_bo5.pth", 0.43328767), ("2nd_argmax_16_4_1073741824_bo5.pth", 0.99830735)]:
    params = load(name)
    for rule, ang, s in [("gm", 0.5, None)]:
        E.RULE, E.MAX_ANGLE, E.GM_S = rule, ang, s
        t0 = time.time()
        acc, st = E.exact_accuracy(params)
        print(f"{name[:18]} {rule} adaptive angle {ang}: {acc:.8f} (ref {ref:.8f}) diff {acc - ref:+.1e} ops {st['ops']:.3g} (quad {st['ops']-st['ops_enum']:.3g}) ({time.time() - t0:.1f}s)", flush=True)
