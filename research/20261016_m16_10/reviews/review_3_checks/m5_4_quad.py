"""Review 3 check: is M5,4's gap to Monte Carlo quadrature or lost mass? Exact with the default rule (centroid on narrow
pieces), with degree-7 Grundmann-Moller on every piece, and with tighter splitting; total mass and direct correct mass."""
import algzoo.exact_nd as E
from algzoo.core import load

p = load("2nd_argmax_5_4_1073741824_bo5.pth")
MC = 0.7456716   # pooled: table_exact_len4.log 0.74567132 +- 6.6e-6 and m5_4_check.log 0.74567229 +- 9.4e-6
for label, gs, ang in [("default", None, 0.5), ("GM degree 7 everywhere", 3, 0.5), ("GM degree 7, split to 0.25 rad", 3, 0.25)]:
    E.GM_S, E.MAX_ANGLE = gs, ang
    acc, st = E.exact_accuracy(p, return_total=True)
    print(f"{label:32s} 1-wrong {acc:.8f} (diff {acc - MC:+.1e}); direct correct {st['correct_direct']:.8f} "
          f"(diff {st['correct_direct'] - MC:+.1e}); total mass {st['total']:.8f}")
