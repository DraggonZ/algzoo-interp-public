"""Neighbour separation. (1) Per unit of the second-largest at s: margin own - (s-1) and own - (s+1), all paths
and the hiding-difference path's part (exact, real on/off pattern). (2) S4 / S7 / S3 mistakes classified."""
from n3lib import *
from sub import quantities, fit_cells, run_mod

X = lists(41, 150_000); S, P, H, JP, JH = jac(X); G = back(P); n = len(X)
o = X.argsort(1); L, Sd = o[:, -1], o[:, -2]
JHs = np.concatenate([np.zeros_like(JH[:, :1]), JH[:, :-1]], 1)
gapJ = WH[3, 1] * JHs[:, :, 1] + WH[3, 8] * JHs[:, :, 8]
tot = np.einsum("ki,bip->bkp", WO, JH[:, 9]); pg = np.einsum("btk,btp->bkp", G[:, :, :, 3], gapJ)
srec = np.array([X[i, Sd[i]] > max(0, X[i, :Sd[i]].max(initial=-9)) for i in range(n)])
for kind, sel in (("kept", Sd > L), ("revealed", (Sd < L) & srec)):
    print(f"\n{kind} second-largest: per unit, margin over the earlier and later neighbour (all paths | hiding-difference path)")
    for s in range(10):
        m = sel & (Sd == s)
        if m.sum() < 500: continue
        def mg(A, j): return (A[m, s, s] - A[m, j, s]).mean() if 0 <= j < 10 else np.nan
        print(f"  s={s}: own-(s-1) {mg(tot, s - 1):6.1f} | {mg(pg, s - 1):6.1f}     own-(s+1) {mg(tot, s + 1):6.1f} | {mg(pg, s + 1):6.1f}     n={m.sum()}")

Xf, Xt = lists(21, 200_000), lists(42, 200_000); Qf, Qt = quantities(Xf), quantities(Xt)
_, Pf, Hf, *_ = jac(Xf)
Dreal = np.zeros_like(Pf[:, :, 3]); Dreal[:, 1:] = WH[3, 1] * Hf[:, :-1, 1] + WH[3, 8] * Hf[:, :-1, 8]
import sub
_feats = sub.feats
def feats2(kind, X_, Q, t):
    if kind == "m2": return Q[2][:, t, 1:2]
    return _feats(kind, X_, Q, t)
sub.feats = feats2
S0 = run(Xt)[0]; yt = Xt.argsort(1)[:, -2]; lt = Xt.argsort(1)[:, -1]; tt = Xt.argsort(1)[:, -3]
print("\nnew mistakes (network right, variant wrong): share naming the position one earlier, one later, the largest")
for nm, kind in (("S3 per-cell mean", "none"), ("S4 largest only", "m1"), ("S7 second-largest only", "m2"), ("S1 gap formula", "gap")):
    Sx = run_mod(Xt, Qt, "D", fit_cells(Dreal, Xf, Qf, kind, range(1, 10), 1), kind); a = Sx.argmax(1)
    new = (S0.argmax(1) == yt) & (a != yt)
    e, l_, lg = (a[new] == yt[new] - 1).mean(), (a[new] == yt[new] + 1).mean(), (a[new] == lt[new]).mean()
    print(f"  {nm:24s} acc {(a == yt).mean() * 100:5.1f}  new mistakes {new.sum():6d}: one earlier {e * 100:4.0f}%  one later {l_ * 100:4.0f}%  the largest {lg * 100:4.0f}%")
own = S0.argmax(1) != yt
print(f"  network's own mistakes: one earlier {(S0.argmax(1)[own] == yt[own] - 1).mean() * 100:4.0f}%  one later {(S0.argmax(1)[own] == yt[own] + 1).mean() * 100:4.0f}%")
