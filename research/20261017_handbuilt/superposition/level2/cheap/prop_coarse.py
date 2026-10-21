"""prop4 (the propagation agent's model, imported unchanged) at coarse value grids: accuracy and Gaussian transitions."""
import sys, time, numpy as np, torch
sys.path.insert(0, "../switching"); sys.path.insert(0, "../../../family")
from prop4 import Prop
from common import trained, ARC
tr = np.load("../../../hilton/propagate/across.npy")[:, 0]
nets = [("ARC", ARC, tr[0]), ("114", trained(114), tr[15]), ("203", trained(203), tr[20])]
for name, W, t in nets:
    Wn = [w.cpu().numpy() for w in W]
    for G in [int(g) for g in sys.argv[1:]]:
        t0 = time.time(); P = Prop(Wn, G=G, lookahead="auto", chunk=1_000_000); acc = P.run()
        print(f"{name} G {G}: {100*acc:.2f}% (true {100*t:.2f})  transitions {P.npairs:.3g}  final Gaussians {P.nfinal:.3g}  {time.time()-t0:.0f}s  peak {torch.cuda.max_memory_allocated()/1e9:.1f} GB", flush=True)
