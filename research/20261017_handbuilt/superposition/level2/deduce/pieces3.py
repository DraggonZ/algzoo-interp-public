"""Robustness of the piece count: count distinct decision-relevant maps, the lead of the second over the third,
d = J[second] - J[third] (a 10-vector, scores difference = d . x), normalised by its own length, rounded to a step of p."""
import sys; sys.path.insert(0, "../../../family")
import numpy as np, torch
from collections import Counter
from common import *
from algzoo.handbuilt_v19 import build_v19
from algzoo.handbuilt_v21 import build_v21
g = torch.Generator(device=DEV).manual_seed(7); N = 200_000; X = torch.randn(N, 10, device=DEV, dtype=dt, generator=g)
l, s, t = top3(X); n = torch.arange(N, device=DEV)
print(f"{N:,} random lists; distinct second-over-third lead maps (unit length) after rounding to step p; in brackets: maps covering half the lists")
for name, W in [("ARC (trained, 16)", ARC), ("trained seed 111", trained(111)), ("hand-built v19 kappa 0.5", T(build_v19(kappa=0.5).params)), ("hand-built v21 (28)", T(build_v21()))]:
    J = jac(W, X); d = J[n, s] - J[n, t]; d = d / d.norm(dim=1, keepdim=True); r = []
    for p in (0.01, 0.03, 0.1, 0.3):
        q = torch.round(d / p).to(torch.int64).cpu().numpy(); c = Counter(map(bytes, q)); cnt = np.array(sorted(c.values(), reverse=True))
        r.append(f"p {p:g}: {len(c):7,} ({np.searchsorted(np.cumsum(cnt) / N, 0.5) + 1:,})")
    print(f"{name:26s} " + " | ".join(r), flush=True)
