"""Linear pieces at the precision the decision needs. On each list scores = J x exactly (J: 10x10, set by the on/off
sequence). The decision only uses differences of scores, so take D = J - J[second row] ... simpler: J rows minus their
mean over outputs. Count distinct J after rounding each entry to a step of p x (typical size of J), for several p.
Compare ARC with hand-built and other trained networks."""
import sys; sys.path.insert(0, "../../../family")
import numpy as np, torch
from collections import Counter
from common import *
from algzoo.core import load
import importlib
def nets():
    yield "ARC (trained, 16)", ARC
    for s in (100, 111, 209): yield f"trained seed {s} (16)", trained(s)
    M = "/data/output/research/algzoo/models/extra/"
    for f in ("2nd_argmax_32_10_1073741824_bo5",): yield "trained 32 neurons", T(load(M + f + ".pth"))
    from algzoo.handbuilt_v19 import build_v19
    from algzoo.handbuilt_v21 import build_v21
    yield "hand-built v19 kappa 0.5 (16)", T(build_v19(kappa=0.5).params)
    yield "hand-built v21 (28)", T(build_v21())
g = torch.Generator(device=DEV).manual_seed(7); N = 200_000
X = torch.randn(N, 10, device=DEV, dtype=dt, generator=g)
print(f"{N:,} random lists; distinct linear maps J after rounding entries to p x RMS(J) (exact = distinct on/off sequences)")
for name, W in nets():
    J = jac(W, X).reshape(N, -1); scale = J.pow(2).mean().sqrt().item(); out = []
    acc = (scores(W, X).argmax(1) == X.argsort(1, descending=True)[:, 1]).double().mean().item()
    for p in (0.0001, 0.01, 0.03, 0.1):
        q = torch.round(J / (p * scale)).to(torch.int64).cpu().numpy()
        c = Counter(map(bytes, q)); cnt = np.array(sorted(c.values(), reverse=True)); cum = np.cumsum(cnt) / N
        out.append(f"p {p:g}: {len(c):7,} ({np.searchsorted(cum, 0.5) + 1:,} cover half)")
    print(f"{name:24s} acc {acc:.3f} | " + " | ".join(out), flush=True)
