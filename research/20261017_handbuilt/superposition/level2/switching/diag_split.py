"""How many pieces the neuron splits create per step, for a few thresholds (ARC's weights)."""
import sys, time, numpy as np
from algzoo.core import example
from prop import Prop
W = [np.asarray(w, float) for w in example()]
eps, wmin, T = float(sys.argv[1]), float(sys.argv[2]), int(sys.argv[3])
t0 = time.time(); P = Prop(W, eps=eps, wmin=wmin, verbose=True)
c = {(-1, -1, ()): (1.0, np.zeros(18), np.zeros((18, 18)))}
for t in range(T): c = P.step(c, t); print(f"  {time.time()-t0:.0f}s", flush=True)
if T == 10: print(f"accuracy {100*P.accuracy(c):.2f}%")
