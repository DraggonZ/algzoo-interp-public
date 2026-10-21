"""Cost of the frozen estimator on ARC against sampling at the same error.
Counts: Gaussian transitions (one source node x one value of the new number x one list event; each is a 16-dim mean
and 16x16 covariance update, about 1.6k floating-point operations, plus a 16x16x16 covariance push per source node),
final integrals (one 9-dim Gaussian orthant per (node, last value) pair, 512 quasi-random points, twice: for the
answer and for the largest), the first pass that picks the look-ahead neurons. One network run on one list is
10 x (2 x 16 x 16 + 2 x 16) + 2 x 10 x 16 = 5,760 operations. Sampling's standard error is sqrt(p(1-p)/n)."""
import sys, time, numpy as np
from algzoo.core import example
from prop4 import Prop
W = [np.asarray(w, float) for w in example()]
RUN = 10 * (2 * 256 + 32) + 2 * 160
p = 0.9532
for G in [int(g) for g in sys.argv[1:]]:
    t0 = time.time(); P = Prop(W, G=G, lookahead="auto", chunk=2_000_000); acc = P.run(); wall = time.time() - t0
    nsrc = P.npairs / (3 * G)                                         # source-node visits (both passes)
    ops = P.npairs * 1600 + nsrc * 2 * 2 * 4096 + P.nfinal * 2 * 512 * (81 + 9)
    print(f"G {G}: predicted {100*acc:.2f}%; Gaussian transitions {P.npairs:.3g}, source-node visits {nsrc:.3g}, final orthant integrals {2*P.nfinal:.3g}; "
          f"about {ops:.2g} operations = {ops/RUN:.2g} network runs; wall {wall:.0f}s on the GPU", flush=True)
    for err in (0.0065, 0.003, 0.001, 0.0005):
        n = p * (1 - p) / err ** 2
        print(f"     sampling with standard error {100*err:.2f} points needs {n:.3g} lists ({n*RUN:.2g} operations): the estimator costs {ops/(n*RUN):.2g} times that")
