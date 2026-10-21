"""What does the keep gate write after a kept number? Lists: fillers -0.5, the largest 1.5 at position i, a kept number
s at position j > i. Final scores of the largest's and the kept number's positions, intact and with the keep gate's
pushes removed one step after s (its value at step j+1 set to 0 for its receivers; readout unchanged unless j = 8).
Also the hiding neurons' values at step j (how many of them s switches off) and neuron 5's value at j+1."""
import numpy as np
from gatelib import WH, run


def scores(x, j, silence):
    def hook(t, hp, pre, h):
        if silence and t == j + 2:          # receivers of neuron 5's value at step j+1
            return np.maximum(pre - np.outer(hp[:, 5], WH[:, 5]), 0)
        return h
    L, P, H = run(x, hook=hook)
    return L[0], H[0]


print("i j    s   | hiding 1,6,7,8 at step j      n5(j+1) | intact: score L  score s  best other | gate silenced: score L  score s  best other")
for (i, j) in ((1, 3), (2, 5), (0, 6)):
    for s in (0.5, 0.9, 1.2, 1.4, 1.48):
        x = np.full((1, 10), -0.5); x[0, i] = 1.5; x[0, j] = s
        L0, H0 = scores(x, j, False); L1, H1 = scores(x, j, True)
        o0 = np.delete(L0, [i, j]).max(); o1 = np.delete(L1, [i, j]).max()
        print(f"{i} {j}  {s:4.2f}  | " + " ".join(f"{v:6.2f}" for v in H0[j, [1, 6, 7, 8]]) + f"   {H0[j + 1, 5]:6.2f}  |"
              f"  {L0[i]:7.1f}  {L0[j]:7.1f}  {o0:7.1f}   |  {L1[i]:7.1f}  {L1[j]:7.1f}  {o1:7.1f}")
