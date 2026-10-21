"""Grid centred on the third (split at 0, where the model switches from three to nine leads) against the m2-centred grid."""
from lib import *
from estimate import estimate
W = load_net("ARC")
for model in ["mean01", "two"]:
    for kind in ["split-gl", "split3-gl"]:
        res = [estimate(W, n, n, model, kind=kind) for n in (12, 16, 24, 32)]
        print(f"{model:6s} {kind:9s} n=12,16,24,32: " + "  ".join(f"{100*r['acc']:.3f}% (mass {r['mass']:.4f})" for r in res), flush=True)
