"""First-order (attribution-patching) prediction of v21's spoiled-timer scores (v21_graded.py), with the backward pass
through v21's own run; group substituted, rest as built."""
import numpy as np
exec(open("v21_graded.py").read().split("H = simlib.run")[0])
L, Pre, H = simlib.run(X, params=P); r = np.arange(B); k = L.argmax(1); j = np.argsort(L, 1)[:, -2]; lead = L[r, k] - L[r, j]
g = P[2][k] - P[2][j]; Gt = np.zeros_like(H)
for t in range(9, -1, -1):
    Gt[:, t] = g; g = (g * (Pre[:, t] > 0)) @ P[1]
cases = {"exact timer rules": timer(), "misreading: height from the largest": timer(height="largest"), "misreading: flip left out": timer(flip=False),
         "slightly off: 45.5 degrees": timer(deg45=45.5), "slightly off: wave 2% too large": timer(gain=1.02), "slightly off: store 2% too large": {I["S"]: 1.02 * H[:, :, I["S"]]}}
for lab, sims in cases.items():
    meas = np.mean(simlib.run(X, sims, params=P)[0].argmax(1) == y)
    dl = sum((Gt[:, :, n] * (v - H[:, :, n])).sum(1) for n, v in sims.items()); pred = np.mean(np.where(lead + dl < 0, j == y, k == y))
    print(f"{lab:38s}: measured {meas:.3f}; first-order predicted {pred:.3f}", flush=True)
