"""Which grid converges fastest? Old 0/1 answer (cheap) and the 'two' model on ARC; converged old-grid value 95.32%."""
import numpy as np
from lib import *
from estimate import estimate
W = load_net("ARC")
for model in ["mean01", "two"]:
    for kind in ["old", "split", "split-gl"]:
        res = [estimate(W, n, n, model, kind=kind) for n in (16, 24, 32, 48)]
        print(f"{model:6s} {kind:9s} n=16,24,32,48: " + "  ".join(f"{100*r['acc']:.3f}% (mass {r['mass']:.4f})" for r in res), flush=True)
