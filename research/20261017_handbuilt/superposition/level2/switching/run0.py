"""First test: propagate ARC's own weights (no gap), regime = positions of the largest and second-largest so far."""
import time, numpy as np
from algzoo.core import example
from prop import Prop
W = [np.asarray(w, float) for w in example()]
t0 = time.time(); P = Prop(W); acc = P.run()
for r in P.log: print("step %d: %d regimes, %d pieces, %d splits, mass %.6f" % r)
print(f"predicted accuracy {100*acc:.2f}%  (true 95.32%)  {time.time()-t0:.0f}s")
