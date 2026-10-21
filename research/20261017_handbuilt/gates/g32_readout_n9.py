"""The late gate's readout and lists with a record at step 8, defined two ways (200,000 lists, seed 2 as g02_jobs.py):
'record at 8' = x_8 beat every earlier number (it may be overtaken at step 9); 'largest at 8' = x_8 is the largest."""
import numpy as np
from gatelib import WI, WH, WO, lists, run, acc_by_kind

X = lists(2); L0 = run(X)[0]
wo = WO.copy(); wo[:, 9] = 0; L1 = run(X, params=(WI, WH, wo))[0]
K = ["all", "record at 8", "L at 8", "record at 9", "L at 9"]
a0, a1 = acc_by_kind(X, L0, K), acc_by_kind(X, L1, K)
for k in K:
    print(f"{k:12s} ({a0[k][1]:6d} lists): intact {100 * a0[k][0]:.1f}%, late gate readout removed {100 * a1[k][0]:.1f}%")
