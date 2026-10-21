"""Is ARC's near-tie decision predictable from a few list features? Target: D / x_S (D = score_S - score_T, exact).
Features (nested): gap (x_S - x_T)/x_S only; + positions of L, S, T (one-hot); + x_L / x_S; + the seven smaller
numbers (sorted by position, / x_S). Linear least squares and gradient-boosted trees (CPU, 4 threads); R^2 and
agreement of sign(D) on held-out lists. Also: the same for trained 209 and for ARC's D predicted by trained 209's
D (do two networks have the same leftover?)."""
import os; os.environ.setdefault("OMP_NUM_THREADS", "4")
import numpy as np, torch
from sklearn.ensemble import HistGradientBoostingRegressor
from common import *
if __name__ == "__main__":
    X = near_lists(240000, seed=1); L, S, Tq = top3(X); ar = torch.arange(len(X), device=DEV)
    xs = X[ar, S]
    D = {nm: ((lambda Sc: (Sc[ar, S] - Sc[ar, Tq]) / xs)(scores(W, X))).cpu().numpy() for nm, W in [("ARC", ARC), ("209", trained(209)), ("111", trained(111))]}
    oh = lambda p: np.eye(10)[p.cpu().numpy()]
    gap = ((xs - X[ar, Tq]) / xs).cpu().numpy()[:, None]; xl = (X[ar, L] / xs).cpu().numpy()[:, None]
    rest = X.clone(); rest[ar, L] = 0; rest[ar, S] = 0; rest[ar, Tq] = 0; rest = (rest / xs[:, None]).cpu().numpy()
    sets = {"gap": gap, "+ positions": np.c_[gap, oh(L), oh(S), oh(Tq)], "+ largest": np.c_[gap, oh(L), oh(S), oh(Tq), xl],
            "+ the seven smaller": np.c_[gap, oh(L), oh(S), oh(Tq), xl, rest]}
    tr, te = slice(0, 200000), slice(200000, None)
    for nm in ("ARC", "209"):
        y = D[nm]; print(f"== {nm}: wrong {100*(y[te]<0).mean():.1f}% of held-out near ties")
        for sn, F in sets.items():
            A = np.c_[F, np.ones(len(F))]; c = np.linalg.lstsq(A[tr], y[tr], rcond=None)[0]; pl = A[te] @ c
            gb = HistGradientBoostingRegressor(max_iter=300, max_leaf_nodes=63).fit(F[tr], y[tr]); pg = gb.predict(F[te])
            r2 = lambda p: 1 - ((y[te] - p) ** 2).mean() / y[te].var()
            print(f"  {sn:22s} linear R2 {r2(pl):.2f} sign agree {100*((pl>0)==(y[te]>0)).mean():.1f}% | trees R2 {r2(pg):.2f} sign agree {100*((pg>0)==(y[te]>0)).mean():.1f}%")
    for a, b in (("ARC", "209"), ("ARC", "111"), ("209", "111")):
        print(f"{a} vs {b}: correlation of D {np.corrcoef(D[a], D[b])[0,1]:.2f}; same sign {100*((D[a]>0)==(D[b]>0)).mean():.1f}%; both wrong {100*((D[a]<0)&(D[b]<0)).mean():.1f}% (independent {100*(D[a]<0).mean()*(D[b]<0).mean():.1f}%)")
