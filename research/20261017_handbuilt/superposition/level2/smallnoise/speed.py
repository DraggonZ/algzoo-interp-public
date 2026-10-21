import sys, time, lib
from lib import *
from estimate import estimate
W = load_net("ARC")
for fast in [torch.float32, torch.float64]:
    lib.FAST = fast; t0 = time.time(); r = estimate(W, 12, 12, "fourth", 2); torch.cuda.synchronize()
    print(fast, f"{100*r['acc']:.4f}", f"{time.time()-t0:.1f}s", flush=True)
