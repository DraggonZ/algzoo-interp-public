"""Match the leak to ARC's measured net effect, not its raw size. v19 (kappa 0.5) with readout rows mixed by
I + s (K - I), K = ARC's crosstalk (crosstalk.measure). For each s: mean |effect| on the near-tie decision per unit of a
kept positive number before the largest and after it, and of negative numbers (trace_where's measure, in votes);
ARC: 0.08-0.12 before, 0.03-0.05 after, under 0.01 for negatives. Then accuracy and mistakes for the s values."""
import numpy as np, torch
from common import *
from algzoo.handbuilt_v19 import build_v19
from crosstalk import measure
import score
def effects(W, X, L, S, Tq):
    ar = torch.arange(len(X), device=DEV); pos = torch.arange(10, device=DEV)[None].expand(len(X), 10)
    rest = torch.ones(len(X), 10, dtype=torch.bool, device=DEV)
    for p in (L, S, Tq): rest[ar, p] = False
    J = jac(W, X); vote = 0.5 * (J[ar, S, S] + J[ar, Tq, Tq]).abs().median(); e = ((J[ar, S, :] - J[ar, Tq, :]) / vote).abs()
    after = pos > L[:, None]; P = X > 0
    return [e[rest & ~after & P].mean().item(), e[rest & after & P].mean().item(), e[rest & ~P].mean().item()]
if __name__ == "__main__":
    K = measure(); I = np.eye(10); wi, wh, wo = build_v19(kappa=0.5).params
    X = near_lists(30000, seed=4); L, S, Tq = top3(X)
    print(f"ARC: |effect| per unit: positive before L {effects(ARC, X, L, S, Tq)[0]:.3f}, after L {effects(ARC, X, L, S, Tq)[1]:.3f}, negative {effects(ARC, X, L, S, Tq)[2]:.3f}")
    models = {}
    for s in (0.0, 0.02, 0.05, 0.1, 0.2):
        W = T([wi, wh, (I + s * (K - I)) @ wo]); e = effects(W, X, L, S, Tq)
        print(f"s {s}: positive before L {e[0]:.3f}, after L {e[1]:.3f}, negative {e[2]:.3f}")
        models[f"v19 + leak s {s}"] = [W]
    score.report(models)
