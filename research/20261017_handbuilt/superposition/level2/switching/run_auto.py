"""ARC with the look-ahead neurons chosen automatically (no role knowledge), at grid sizes from argv."""
import sys, time, numpy as np
from algzoo.core import example
from prop4 import Prop
W = [np.asarray(w, float) for w in example()]
for G in [int(g) for g in sys.argv[1:]]:
    t0 = time.time(); P = Prop(W, G=G, lookahead="auto", chunk=2_000_000); acc = P.run()
    print(f"G {G}: misplaced switch probability per neuron (first pass): " + " ".join(f"{j}:{d:.2f}" for j, d in sorted(enumerate(P.disc_all), key=lambda a: -a[1])[:6]))
    print(f"G {G}: look-ahead chosen {P.la}; predicted {100*acc:.2f}% (true 95.32%), mistakes naming the largest {100*P.share_L:.1f}%  {time.time()-t0:.0f}s", flush=True)
