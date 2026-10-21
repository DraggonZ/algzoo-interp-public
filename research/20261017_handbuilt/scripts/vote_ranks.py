"""Which inputs decide the answer? Accuracy when only the pushes of the m largest inputs are kept, and the
second-vs-third margin split by input rank."""
import numpy as np, torch as th
from vote_ledger import ledger

th.manual_seed(1)
X = th.randn(400_000, 10, device="cuda", dtype=th.float64)
S, V = ledger(X)
order = X.argsort(1, descending=True)              # order[:, r] = position of rank r (0 = largest)
rank = order.argsort(1)                            # rank of each position
tgt = order[:, 1]
print("network accuracy", (S.argmax(1) == tgt).double().mean().item())
for m in range(1, 11):
    keep = (rank < m).double()[:, :, None]
    print(f"pushes of the {m:2d} largest only: accuracy {((V * keep).sum(1).argmax(1) == tgt).double().mean().item():.4f}")

r = th.arange(len(X), device="cuda")
s2, s3 = order[:, 1], order[:, 2]
M = V[r, :, s2] - V[r, :, s3]                      # (n, t): input t's push on score(second) - score(third)
Mr = th.gather(M, 1, order)                        # by rank
print("\nsecond-minus-third margin, mean push by input rank (0 = largest):")
print(np.round(Mr.mean(0).cpu().numpy(), 1))
print("median |push| by rank:", np.round(Mr.abs().median(0).values.cpu().numpy(), 1))
wrong = S.argmax(1) == s3
print(f"\nlists answered 'third' ({wrong.double().mean().item():.4f}): mean push by rank")
print(np.round(Mr[wrong].mean(0).cpu().numpy(), 1))
