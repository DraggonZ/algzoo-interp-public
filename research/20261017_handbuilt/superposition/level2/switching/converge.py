"""Frozen estimator on ARC: convergence with grid size, and dependence on the tie spacing at the largest grid.
Settings (frozen before the held-out checks): look-ahead neurons chosen automatically (3), tie spacing 1/3 of the cell
width (expected gap of two values uniform in one cell), cells by the top-three density, last number kept exact."""
import sys, time, numpy as np
from algzoo.core import example
from prop4 import Prop
W = [np.asarray(w, float) for w in example()]
for G, tf in [(20, 1/3), (40, 1/3), (60, 1/3), (80, 1/3), (80, 1/4), (80, 1/2)]:
    t0 = time.time(); P = Prop(W, G=G, lookahead="auto", tie_frac=tf, chunk=2_000_000); acc = P.run()
    print(f"G {G}, tie spacing {tf:.3f} of a cell: look-ahead {P.la}; predicted {100*acc:.2f}% (true 95.32%), mistakes naming the largest {100*P.share_L:.1f}% (true 20.3%)  {time.time()-t0:.0f}s", flush=True)
