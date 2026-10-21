"""Independent check of the helper's v23-from-v19: accuracy on 1,000,000 fresh lists, and the nudge battery beside ARC,
v19 and v23."""
import sys; sys.path.insert(0, "/data/output/research/algzoo/20261017_handbuilt/family")
import numpy as np, torch
from battery import battery, compare, hb_groups, ARC, ARC_GROUPS, T
from common import scores, top3, DEV, dt
from algzoo.handbuilt_v19 import build_v19
from algzoo.handbuilt_v23 import build_v23
from algzoo.handbuilt_v23_from_v19 import build_v23f
if __name__ == "__main__":
    f = build_v23f()
    g = torch.Generator(device=DEV).manual_seed(424242); X = torch.randn(1_000_000, 10, device=DEV, dtype=dt, generator=g)
    print(f"v23 from v19, accuracy on 1,000,000 fresh lists: {100*(scores(T(f.params), X).argmax(1) == top3(X)[1]).double().mean().item():.2f}%")
    nets = {"ARC": (ARC, ARC_GROUPS), "v19": (T(build_v19(kappa=0.5).params), hb_groups(build_v19(kappa=0.5))),
            "v23 from v19": (T(f.params), hb_groups(f)), "v23": (T(build_v23().params), hb_groups(build_v23()))}
    res = {n: battery(W, G) for n, (W, G) in nets.items()}; compare(res)
