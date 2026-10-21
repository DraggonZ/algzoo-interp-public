"""What decides ARC's near ties between the second (S) and third (T)? Exact split of D = score_S - score_T into:
own votes J[S,S] x_S - J[T,T] x_T; the largest's leak (J[S,L] - J[T,L]) x_L; cross terms J[S,T] x_T - J[T,S] x_S;
the seven smaller numbers. Random lists, third within 5% of the second, second positive. For ARC and a trained
sibling: share of wrong (D < 0) by group, each part's median in units of x_S, and the variance of D each part
explains (covariance share) - split by whether T arrives after S."""
import numpy as np, torch, sys
from common import *
if __name__ == "__main__":
    X = near_lists(200000); L, S, Tq = top3(X); ar = torch.arange(len(X), device=DEV)
    for name, W in [("ARC", ARC), ("trained 111", trained(111)), ("trained 209", trained(209))]:
        J = jac(W, X); xs, xt, xl = X[ar, S], X[ar, Tq], X[ar, L]
        own = J[ar, S, S] * xs - J[ar, Tq, Tq] * xt; leak = (J[ar, S, L] - J[ar, Tq, L]) * xl
        cross = J[ar, S, Tq] * xt - J[ar, Tq, S] * xs
        C = (J[ar, S, :] - J[ar, Tq, :]) * X; rest = C.sum(1) - (own + leak + cross)
        Dd = own + leak + cross + rest
        print(f"== {name}: wrong (third beats second) {100*(Dd<0).double().mean():.1f}% of these lists; median own vote per unit J[S,S] {J[ar,S,S].median():.1f}")
        for gname, m in [("all", torch.ones_like(S, dtype=torch.bool)), ("T after S", Tq > S), ("T before S", Tq < S), ("T at 9", Tq == 9), ("S at 9", S == 9)]:
            parts = {"own": own[m], "leak L": leak[m], "cross": cross[m], "rest": rest[m]}; d = Dd[m]; vs = d.var()
            print(f"  {gname:11s} n {m.sum().item():6d} wrong {100*(d<0).double().mean():5.1f}% | median/x_S: " +
                  "  ".join(f"{k} {(v / xs[m]).median():+6.2f}" for k, v in parts.items()) + " | var share: " +
                  "  ".join(f"{k} {((v - v.mean()) * (d - d.mean())).mean() / vs:+.2f}" for k, v in parts.items()))
        # own vote sizes: J[p,p] per unit by position for S and T roles
        print("  own vote per unit by position (median J[p,p] when p holds S or T): " + " ".join(f"{p}:{J[ar, S, S][S == p].median():.1f}" for p in range(10)))
