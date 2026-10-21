"""Shape of the near-tie leftover: share of near ties (third within 5% of second, second > 0) where the third wins,
by the largest's size relative to the second (x_L / x_S) and by arrival order of L, S, T. ARC, trained 111, 209,
mean of 5 trained (103, 111, 115, 209, 210), v19 (kappa 0.5)."""
import numpy as np, torch
from common import *
from algzoo.handbuilt_v19 import build_v19
if __name__ == "__main__":
    X = near_lists(600000, seed=2); L, S, Tq = top3(X); ar = torch.arange(len(X), device=DEV)
    r = (X[ar, L] / X[ar, S]); ords = ["".join("LST"[j] for j in np.argsort(p)) for p in torch.stack([L, S, Tq], 1).cpu().numpy()]
    ords = np.array(ords); rb = np.digitize(r.cpu().numpy(), [1.1, 1.25, 1.5, 2.0, 3.0])
    nets = {"ARC": [ARC], "111": [trained(111)], "209": [trained(209)], "5 trained": [trained(s) for s in (103, 111, 115, 209, 210)], "v19": [T(P(build_v19(kappa=0.5)))]}
    win = {k: np.mean([(scores(W, X).argmax(1) == Tq).double().cpu().numpy() for W in v], 0) for k, v in nets.items()}
    print("third wins, by x_L / x_S:"); print(f"{'bin':12s} {'share':>6s} " + " ".join(f"{k:>9s}" for k in nets))
    for b, nm in enumerate(["<1.1", "1.1-1.25", "1.25-1.5", "1.5-2", "2-3", ">3"]):
        m = rb == b; print(f"{nm:12s} {100*m.mean():5.1f}% " + " ".join(f"{100*win[k][m].mean():8.1f}%" for k in nets))
    print("third wins, by arrival order:")
    for o in ["LST", "LTS", "SLT", "TLS", "STL", "TSL"]:
        m = ords == o; print(f"{o:12s} {100*m.mean():5.1f}% " + " ".join(f"{100*win[k][m].mean():8.1f}%" for k in nets))
    print("third wins, order x ratio (ARC / 5 trained / v19):")
    for o in ["LST", "LTS", "SLT", "TLS", "STL", "TSL"]:
        print(f"  {o}: " + "  ".join(f"{nm} {100*win['ARC'][(ords==o)&(rb==b)].mean():4.0f}/{100*win['5 trained'][(ords==o)&(rb==b)].mean():2.0f}/{100*win['v19'][(ords==o)&(rb==b)].mean():2.0f}" for b, nm in enumerate(["<1.1", "1.1-1.25", "1.25-1.5", "1.5-2", "2-3", ">3"])))
