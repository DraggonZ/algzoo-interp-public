"""Grid convergence of the new estimates on ARC's network (true 95.319%, 20M lists; 95.312% on 20M other lists),
on the split Gauss-Legendre grid (grids.py), n points per axis."""
import time, numpy as np
from lib import *
from estimate import estimate
W = load_net("ARC")
for model, n, Q4 in [("fourth", 8, 2), ("fourth", 12, 2), ("fourth", 16, 2), ("fourth", 12, 4), ("fourth", 16, 4), ("fourth", 24, 2), ("fourth", 16, 8)]:
    t0 = time.time(); r = estimate(W, n, n, model, Q4)
    print(f"{model:6s} grid {n}^3 Q4={Q4}: {100*r['acc']:.3f}%  mistakes naming largest/third/smaller {np.round(100*r['split'],2)}  "
          f"mass {r['mass']:.4f}  run-equivalents {r['runs']/1e6:.1f}M  {time.time()-t0:.0f}s", flush=True)
