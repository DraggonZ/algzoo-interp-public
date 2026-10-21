"""Split ARC's cross term at near ties into its two halves, by arrival order. E = earlier of S and T, Lt = later.
J[Lt,E]: the earlier number's effect on the later one's score (a discount by the runner-up if negative);
J[E,Lt]: the later number's effect on the earlier one's score (push-down if negative). Per unit, medians and
spread (10th-90th percentile), by the later number's position. Also own votes J[E,E], J[Lt,Lt]."""
import numpy as np, torch
from common import *
if __name__ == "__main__":
    X = near_lists(200000); L, S, Tq = top3(X); ar = torch.arange(len(X), device=DEV)
    E = torch.minimum(S, Tq); Lt = torch.maximum(S, Tq)
    for name, W in [("ARC", ARC), ("trained 111", trained(111)), ("trained 209", trained(209))]:
        J = jac(W, X); q = lambda v: f"{v.median():6.1f} [{v.quantile(.1):6.1f},{v.quantile(.9):6.1f}]"
        print(f"== {name}   columns: own earlier J[E,E] | own later J[Lt,Lt] | earlier on later J[Lt,E] | later on earlier J[E,Lt] | net later-minus-earlier at a tie (J[Lt,Lt]+J[Lt,E]) - (J[E,E]+J[E,Lt])")
        for p in range(1, 10):
            m = Lt == p
            a, b, c, d = J[ar, E, E][m], J[ar, Lt, Lt][m], J[ar, Lt, E][m], J[ar, E, Lt][m]
            print(f"  later at {p}: {q(a)} | {q(b)} | {q(c)} | {q(d)} | {q((b + c) - (a + d))}  (share wrong when later is T: {100*((X[ar,Tq] > -9) & (Tq == Lt) & m).double().sum().item() and 0:.0f})")
