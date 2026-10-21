"""Propagation version 3 (conditional Gaussian, list values on a grid) on ARC's own weights, at grid sizes G."""
import sys, time, numpy as np
from algzoo.core import example
from prop3 import Prop
W = [np.asarray(w, float) for w in example()]
import json
LA = json.loads(sys.argv[1])
for G in [int(g) for g in sys.argv[2:]]:
    t0 = time.time(); P = Prop(W, G=G, lookahead=LA, verbose=False); acc = P.run()
    print(f"G {G}: predicted accuracy {100*acc:.2f}% lookahead {LA} (true 95.32%), mass {P.mass:.5f}, mistakes naming the largest {100*P.share_L:.1f}%  {time.time()-t0:.0f}s", flush=True)
