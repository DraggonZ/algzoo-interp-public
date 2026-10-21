"""Propagation (prop4) on ARC's own weights at several grid sizes; look-ahead neurons from argv[1]."""
import sys, time, json, numpy as np
from algzoo.core import example
from prop4 import Prop
W = [np.asarray(w, float) for w in example()]
LA = json.loads(sys.argv[1]); KW = json.loads(sys.argv[2]); sys.argv.pop(2)
for G in [int(g) for g in sys.argv[2:]]:
    t0 = time.time(); P = Prop(W, G=G, lookahead=LA, **KW); acc = P.run()
    print(f"G {G}, look-ahead {LA} {KW}: predicted {100*acc:.2f}% (true 95.32%), mass {P.mass:.5f}, mistakes naming the largest {100*P.share_L:.1f}% (true 20%)  {time.time()-t0:.0f}s", flush=True)
