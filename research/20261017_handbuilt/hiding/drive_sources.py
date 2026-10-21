"""How big is each source's push on the four hiding neurons' pre-activations? pre_n(t) = sum_j W[n, j] h_j(t-1) + w_n x_t.
RMS over lists and steps 1-9 of each group's term, and of the group term with its mean over lists (per step) removed."""
import numpy as np
from common import WI, WH, C, GROUPS, lists, run
X = lists(1, 100_000); _, P, H = run(X)
Hp = np.concatenate([np.zeros_like(H[:, :1]), H[:, :-1]], 1)
for n in C:
    out = []
    for g, idx in GROUPS.items():
        term = Hp[:, 1:, idx] @ WH[n, idx]
        out.append(f"{g} {np.sqrt((term ** 2).mean()):5.2f} ({term[:, :].std(0).mean():5.2f})")
    xt = X[:, 1:] * WI[n]
    print(f"neuron {n}: input {np.sqrt((xt ** 2).mean()):5.2f} | " + " | ".join(out))
print("mean activation of each neuron (steps 1-9):", np.round(H[:, 1:].mean((0, 1)), 2))
