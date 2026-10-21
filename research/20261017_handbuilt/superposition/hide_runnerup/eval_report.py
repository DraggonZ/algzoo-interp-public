"""Accuracy on 1,000,000 random lists, and the shared scorer (family/score.py) for v19 kappa 0.5 and v22a."""
import sys, torch
sys.path.insert(0, "../../family")
from common import *
import score
from algzoo.handbuilt_v19 import build_v19
from build_v22a import build_v22a, LINEAR_SHAPE
NETS = {"ARC": ARC, "v19 kappa 0.5 (16)": T(build_v19(kappa=0.5).params), "v22a linear (46)": T(build_v22a(LINEAR_SHAPE).params), "v22a (50)": T(build_v22a().params)}
g = torch.Generator(device=DEV).manual_seed(11); X = torch.randn(1000000, 10, device=DEV, dtype=dt, generator=g)
y = X.argsort(1, descending=True)[:, 1]
for nm, W in NETS.items():
    print(f"{nm:20s} accuracy on 1,000,000 random lists: {100 * (scores(W, X).argmax(1) == y).double().mean():.2f}%  (neurons: {len(W[0])})")
print()
score.report({k: [W] for k, W in NETS.items() if k != "ARC"})
