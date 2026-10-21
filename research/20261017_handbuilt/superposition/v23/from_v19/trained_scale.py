"""v19's 'trained scale' setting (gains 10, vote 10 per unit; robust_twin/sensitivity.py) applied to v23 from v19, with
the loop gain from the same rule (position 0's readout gain 1/ep no larger than the largest other gain, here 10 ->
ep = 0.1) and a smaller one for comparison; then weight noise."""
import torch
from check_basic import T, lists, acc, DEV, dt
from algzoo.handbuilt_v19 import build_v19
from algzoo.handbuilt_v23_from_v19 import build_v23f
TS = dict(g=10, a=10, w=10, c=-9, eps=0.1)

if __name__ == "__main__":
    X = lists(300_000, 4)
    nets = {"v19 trained scale": build_v19(kappa=0.5, lam=10, **TS), "v23f trained scale, ep 0.1": build_v23f(ep=0.1, **TS),
            "v23f trained scale, ep 0.01": build_v23f(ep=0.01, **TS)}
    for name, net in nets.items():
        W0 = T(net); row = [f"clean {100*acc(W0, X):.1f}%"]
        for sd in (0.001, 0.003, 0.01):
            a = []
            for rep in range(2):
                gn = torch.Generator(device=DEV).manual_seed(1000 + rep)
                a.append(acc([w * (1 + sd * torch.randn(w.shape, device=DEV, dtype=dt, generator=gn)) for w in W0], X))
            row.append(f"noise {sd:g}: {100*min(a):.1f}-{100*max(a):.1f}%")
        print(f"{name:30s} " + " | ".join(row), flush=True)
