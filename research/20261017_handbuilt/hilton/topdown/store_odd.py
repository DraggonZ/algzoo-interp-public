"""Lists where, 7 steps after the largest arrived, the store's weight on the largest is near 0: print a few."""
import numpy as np
exec(open("store_jacobian.py").read().split('print("age |')[0])
a = 7
for t in range(7, 10):
    idx = np.where((R[:, t] == t - a) & (Xn[ar, np.maximum(R[:, t], 0)] > 0.3) & (JS[ar, t, np.maximum(R[:, t], 0)] < 5))[0]
    tot = ((R[:, t] == t - a) & (Xn[ar, np.maximum(R[:, t], 0)] > 0.3)).sum()
    print(f"step {t}: {len(idx)} of {tot} lists have weight < 5 on the largest")
    for i in idx[:3]:
        print("   numbers", np.round(Xn[i, :t + 1], 2), "| store", round(S[i, t], 2), "| weights", np.round(JS[i, t, :t + 1], 2))
