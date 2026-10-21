"""Review 2 check: why does exact_nd give M2,5 = 0.30522534 against float64 MC 0.30517429 +- 7.0e-6 (z = +7.3)?
Variant 'total': default settings, also integrate every piece; the total mass must be 1.
Variant 'deg7': degree-7 rule on every piece (NARROW = 0), to test the degree-3 rule on narrow pieces.
Variant 'split': MAX_ANGLE 0.35 instead of 0.5."""
import sys
import time

sys.path.insert(0, "/data/output/research/algzoo/20261016_run2/scripts")
import exact_nd as E  # noqa: E402

from algzoo.core import load  # noqa: E402

variant = sys.argv[1]
name = sys.argv[2] if len(sys.argv) > 2 else "2nd_argmax_2_5_1073741824_bo5.pth"
if variant == "deg7":
    E.NARROW = 0.0
if variant == "split":
    E.MAX_ANGLE = 0.35
t0 = time.time()
acc, st = E.exact_accuracy(load(name), return_total=(variant == "total"))
msg = f"{variant} {name}: exact {acc:.8f} (MC 0.30517429 +- 7.0e-6) ops {st['ops']:.3g} enum {st['ops_enum']:.3g} ({time.time() - t0:.0f}s)"
if "total" in st:
    msg += f" | correct direct {st['correct_direct']:.8f} total mass {st['total']:.8f} (1 - wrong + correct - 1 = {acc - st['correct_direct']:+.2e})"
print(msg, flush=True)
