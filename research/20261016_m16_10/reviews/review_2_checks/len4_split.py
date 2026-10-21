"""Review 2 check: at length 4, how much of exact_nd's cost is enumeration and how much is cone quadrature?"""
import sys
import time

sys.path.insert(0, "/data/output/research/algzoo/20261016_run2/scripts")
import exact_nd as E  # noqa: E402

from algzoo.core import load  # noqa: E402

for name in sys.argv[1:]:
    t0 = time.time()
    acc, st = E.exact_accuracy(load(name))
    print(f"{name}: exact {acc:.8f} regions {st['regions']} wrong pieces {st['wrong_pieces']} ops {st['ops']:.3g}"
          f" enumeration {st['ops_enum']:.3g} ({st['ops_enum'] / st['ops']:.0%}) ({time.time() - t0:.0f}s)", flush=True)
