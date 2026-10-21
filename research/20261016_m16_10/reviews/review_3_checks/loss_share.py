"""Review 3 check: what share of the copies' accuracy loss is two-spike loss (exact Z_2 from z2_check.npz)?"""
import numpy as np

z = np.load("/data/output/research/algzoo/20261016_run2/reports/z2_check.npz", allow_pickle=True)
names = [str(n) for n in z["names"]]
Z, A = z["z2_exact"], z["acc"]
for s in ("0.005", "0.01", "0.02"):
    k = np.array([n.startswith("mult" + s + "_") for n in names])
    zl, tl = (Z[0] - Z[k]).mean(), (A[0] - A[k]).mean()
    print(f"sigma {s}: mean two-spike loss {zl:.4f}, mean true loss {tl:.4f}, share {zl / tl:.2f}")
k = np.array([n.startswith("mult") for n in names])
print("all copies: two-spike loss / true loss", round((Z[0] - Z[k]).sum() / (A[0] - A[k]).sum(), 2))
