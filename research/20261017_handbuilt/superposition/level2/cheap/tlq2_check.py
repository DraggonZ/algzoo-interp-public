"""TLQ2 (grouped) on ARC and two development networks at m = 1..16; how far from the truth, cost, and spread over Sobol seeds."""
import sys, numpy as np, torch, time
sys.path.insert(0, "../../../family")
from common import ARC, trained
import tlq2 as tlq
from linelib import F_LIST
tr = np.load("../../../hilton/propagate/across.npy")[:, 0]
for name, W, t in (("ARC", ARC, tr[0]), ("203", trained(203), tr[20]), ("208", trained(208), tr[25])):
    for m in (1, 4, 16):
        accs = []; t0 = time.time()
        for seed in range(8):
            a, fl = tlq.estimate(W, m, seed); accs.append(a)
        accs = np.array(accs); lists = fl / F_LIST
        samp = 100 * np.sqrt(t * (1 - t) / lists)
        print(f"{name} m {m}: mean over 8 seeds {100*accs.mean():.3f} (true {100*t:.3f}), spread (SD) {100*accs.std():.3f}, RMS error {100*np.sqrt(np.mean((accs-t)**2)):.3f}; "
              f"cost {lists:.0f} lists; sampling SE at that cost {samp:.3f}  {time.time()-t0:.0f}s", flush=True)
