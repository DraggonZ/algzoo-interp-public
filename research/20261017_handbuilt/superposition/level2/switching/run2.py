"""Propagation version 2 on ARC's own weights (no gap): settings from the command line."""
import sys, time, json, numpy as np
from algzoo.core import example
from prop2 import Prop
W = [np.asarray(w, float) for w in example()]
kw = json.loads(sys.argv[1]) if len(sys.argv) > 1 else {}
t0 = time.time(); P = Prop(W, verbose=True, **kw); acc = P.run()
print(f"settings {kw}: predicted accuracy {100*acc:.2f}%  (true 95.32%)  {time.time()-t0:.0f}s")
