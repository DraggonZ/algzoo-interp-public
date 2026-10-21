"""Q1, final state: neuron 3's final value per unit of each number, by the number's rank in the whole list and
position, and by where the largest sits. Exact split (J at step 9)."""
from n3lib import *

X = lists(7, 300_000); S, P, H, JP, JH = jac(X); n = len(X)
J = JH[:, 9, 3]; o = X.argsort(1); L, Sd, Th = o[:, -1], o[:, -2], o[:, -3]; r = np.arange(n)
np.set_printoptions(precision=1, suppress=True, linewidth=160)
print("mean final value of neuron 3:", H[:, 9, 3].mean().round(2), " off at the end:", (P[:, 9, 3] <= 0).mean().round(3))
for nm, idx in (("largest", L), ("second-largest", Sd), ("third", Th)):
    pu = J[r, idx]; print(f"\nper unit of the {nm}, by its position 0-9 (mean over lists):", np.array([pu[idx == p].mean() for p in range(10)]))
print("\nper unit of the second-largest: rows = position of second, columns = position of largest")
M = np.full((10, 10), np.nan)
for s in range(10):
    for l in range(10):
        m = (Sd == s) & (L == l)
        if m.sum() > 300: M[s, l] = J[m, s].mean()
print(M)
print("\nper unit of the largest: rows = position of second, columns = position of largest")
for s in range(10):
    for l in range(10):
        m = (Sd == s) & (L == l)
        if m.sum() > 300: M[s, l] = J[m, l].mean()
print(M)
print("\nshare of variance of neuron 3's final value explained by the shares of largest, second, third, rest:")
c = J * X; parts = {"largest": c[r, L], "second": c[r, Sd], "third": c[r, Th]}; parts["rest"] = c.sum(1) - sum(parts.values())
v = H[:, 9, 3]
for k, p in parts.items(): print(f"   {k:8s} mean {p.mean():6.2f}  cov share {np.cov(p, v)[0, 1] / v.var():5.2f}")
