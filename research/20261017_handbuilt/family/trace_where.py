"""Where do the smaller numbers act on the near-tie decision? At near ties (third within 5% of second, second > 0), for
each smaller number s: (J[S,s] - J[T,s]) / vote, signed so that positive helps the right answer. Grouped by whether s
arrived before or after the largest, and by its distance to the nearer of S and T (1, 2, 3+). Mean and mean absolute.
ARC, trained 209, v19 with traces (eta 0.03)."""
import numpy as np, torch
from common import *
from traces import build
if __name__ == "__main__":
    X = near_lists(40000, seed=4); L, S, Tq = top3(X); ar = torch.arange(len(X), device=DEV)
    pos = torch.arange(10, device=DEV)[None].expand(len(X), 10)
    rest = torch.ones(len(X), 10, dtype=torch.bool, device=DEV)
    for p in (L, S, Tq): rest[ar, p] = False
    after = pos > L[:, None]; dist = torch.minimum((pos - S[:, None]).abs(), (pos - Tq[:, None]).abs()).clamp(max=3)
    sgn = X > 0
    for nm, W in {"ARC": ARC, "trained 209": trained(209), "v19 traces 0.03": T(build(0.03))}.items():
        J = jac(W, X); vote = 0.5 * (J[ar, S, S] + J[ar, Tq, Tq]).abs().median(); e = (J[ar, S, :] - J[ar, Tq, :]) / vote
        print(f"== {nm}  (mean effect per unit / mean |effect| per unit, in votes)")
        for a_nm, a in (("before L", ~after), ("after L", after)):
            for sg_nm, sg in (("neg", ~sgn), ("pos", sgn)):
                cells = []
                for d in (1, 2, 3):
                    m = rest & a & sg & (dist == d)
                    cells.append(f"dist {d}{'+' if d == 3 else ''}: {e[m].mean():+.3f}/{e[m].abs().mean():.3f} (n {m.sum().item()})")
                print(f"  {a_nm:8s} {sg_nm}: " + " | ".join(cells))
