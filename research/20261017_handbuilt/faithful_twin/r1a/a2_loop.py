"""Own accuracy (1M lists, CPU) of v25 with the switch-off (eps 0.1) for a few loop weights w; memory block's modes.
Two hand-picked loop weights: 0.22 (loop modes shrink 0.82 per step, ARC's slowest turning mode) and 0.07 (just above
the band's 0.7 size). Usage: a2_loop.py w1 w2 ..."""
import sys, numpy as np, torch as th
from algzoo.handbuilt_v25 import build_v25
from algzoo.core import forward
X = th.randn(1_000_000, 10, generator=th.Generator().manual_seed(3), dtype=th.float64)
for w in map(float, sys.argv[1:]):
    net = build_v25(eps=0.1, w=w); P = [np.asarray(p, float) for p in net.params]
    acc = (forward(P, X).argmax(-1) == X.argsort(-1)[:, -2]).double().mean().item()
    M = [net.idx[f"mem{k}"] for k in range(8)]; ev = np.linalg.eigvals(P[1][np.ix_(M, M)])
    print(f"w {w}: accuracy {100 * acc:.2f}%  modes size {np.abs(ev).max():.3f}")
