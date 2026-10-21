"""Q1: what neuron 3 holds. Exact split of its pre-activation (and value) by source position, grouped by the
source's rank among the numbers seen so far and by the age of the largest. Also hiding 1, hiding 8, and the
weighted difference 1.20*h1 - 1.07*h8 that dominates neuron 3's input."""
from n3lib import *

X = lists(5, 100_000); S, P, H, JP, JH = jac(X); V = task(X); R = ranks(X)
n = len(X); T = 10
# check the split
err = np.abs(np.einsum("btip,bp->bti", JP, X) - P).max(); print(f"split check: max |sum of shares - pre| = {err:.1e}")
np.set_printoptions(precision=2, suppress=True, linewidth=160)

def table(name, Jsel, value):
    """Jsel (n,T,T): per-unit share of each source at step t. Prints mean share (J*x) and mean per-unit J
    for the largest so far, second so far, third, rest; and the variance of value explained by each group's share."""
    c = Jsel * X[:, None, :]
    print(f"\n=== {name}: mean value by step, and mean share by rank of the source among numbers seen so far")
    print("step  value   largest  second   third    rest   | per unit: largest second third | R2 from (largest+second) share, all shares by rank")
    for t in range(T):
        row = [value[:, t].mean()]; pu = []
        grp = {}
        for k, sel in (("1", R[:, t] == 1), ("2", R[:, t] == 2), ("3", R[:, t] == 3), ("rest", R[:, t] >= 4)):
            s = (c[:, t] * sel).sum(1); grp[k] = s; row.append(s.mean())
            if k != "rest": pu.append((Jsel[:, t] * sel).sum(1)[sel.any(1)].mean() if sel.any() else np.nan)
        v = value[:, t]; r2 = 1 - ((v - grp["1"] - grp["2"]) ** 2).mean() / v.var() if t else np.nan
        r2b = 1 - ((v - grp["1"] - grp["2"] - grp["3"]) ** 2).mean() / v.var() if t > 1 else np.nan
        print(f"{t:4d} " + " ".join(f"{z:7.2f}" for z in row) + "   | " + " ".join(f"{z:6.2f}" for z in pu) + f"   | {r2:5.2f} {r2b:5.2f}")

table("neuron 3 pre-activation", JP[:, :, 3], P[:, :, 3])
table("neuron 3 value", JH[:, :, 3], H[:, :, 3])
d = 1.20 * JH[:, :, 1] - 1.07 * JH[:, :, 8]
table("1.20*h1 - 1.07*h8 (at step t, as it enters neuron 3 at t+1)", d, 1.20 * H[:, :, 1] - 1.07 * H[:, :, 8])
table("hiding 1", JH[:, :, 1], H[:, :, 1]); table("hiding 8", JH[:, :, 8], H[:, :, 8])
print("\noff-rate of neuron 3 by step:", (P[:, :, 3] <= 0).mean(0))
