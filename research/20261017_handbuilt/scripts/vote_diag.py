"""Own-position pushes versus spill. diag: each input pushes only its own position's score, by its exact amount
J[t, t] x_t (centred profile). Spill = everything else."""
import numpy as np, torch as th
from vote_ledger import ledger

th.manual_seed(3)
N = 1_000_000
X = th.randn(N, 10, device="cuda", dtype=th.float64)
S, V = ledger(X)
order = X.argsort(1, descending=True); tgt = order[:, 1]
acc = lambda sc: (sc.argmax(1) == tgt).double().mean().item()
eye = th.eye(10, device="cuda", dtype=X.dtype)
D = (V * eye).sum(2)                                   # (n, t): own-position push
spill = V.sum(1) - D                                   # (n, k): push on k from all other inputs
print("network                       ", acc(S))
print("own-position pushes only      ", acc(D))
print("spill only                    ", acc(spill))
r = th.arange(N, device="cuda")
for name, i in (("largest", 0), ("second", 1), ("third", 2), ("fourth", 3)):
    p = order[:, i]
    d = D[r, p] / X[r, p]; sp = spill[r, p]
    print(f"{name:8s} own push per unit: median {d.median().item():6.1f}  IQR {d.quantile(.25).item():6.1f}..{d.quantile(.75).item():6.1f}"
          f"   spill landing on its position: median {sp.median().item():6.1f}")
