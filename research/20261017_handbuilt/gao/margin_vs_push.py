"""(Second measure added after noticing that pushes common to all positions cancel in the answer: pushes on the
winner's lead over the runner-up.) Why must an explanation be about 1% exact? Compare the decision margin (gap between the top two final scores) with
the size of the pushes that make up the winning score: each neuron's contribution w_o[k, n] * h_n to the score of the
winning position k. If the margin is a small fraction of the pushes, a small relative error in any one push can flip
the answer. Trained network, the twin and the hand-built v7 and v5, 100,000 lists."""
import numpy as np
import simlib
from algzoo.handbuilt_v18 import build_v18

X = simlib.lists(909, 100_000)
P18 = tuple(np.asarray(p, float) for p in build_v18().params)
import sys
sys.path.insert(0, "../scripts")
from handbuilt_v7 import build_v7
from algzoo.handbuilt import build_v5
P7 = tuple(np.asarray(p, float) for p in build_v7().params); P5 = tuple(np.asarray(p, float) for p in build_v5().params)
P5 = (P5[0].reshape(-1), P5[1], P5[2]); P7 = (P7[0].reshape(-1), P7[1], P7[2])
for name, P in (("trained", (simlib.WI, simlib.WH, simlib.WO)), ("twin", P18), ("v7", P7), ("v5", P5)):
    L, _, H = simlib.run(X, params=P); h = H[:, 9]; wo = P[2]
    srt = np.sort(L, 1); margin = srt[:, -1] - srt[:, -2]
    k = L.argmax(1); j = np.argsort(L, 1)[:, -2]
    contrib = wo[k] * h  # (B, 16): each neuron's push on the winning score
    dcontrib = (wo[k] - wo[j]) * h  # each neuron's push on the winner's lead over the runner-up (common pushes cancel)
    dsum = np.abs(dcontrib).sum(1); rd = margin / np.maximum(dsum, 1e-12)
    total_abs = np.abs(contrib).sum(1); biggest = np.abs(contrib).max(1)
    r = margin / total_abs
    print(f"{name:7s}: median margin {np.median(margin):.2f}; median sum of |pushes| on the winning score {np.median(total_abs):.1f}, "
          f"largest single push {np.median(biggest):.1f}; margin / sum of |pushes|: median {np.median(r):.3f}, "
          f"10th percentile {np.percentile(r, 10):.4f}; share of lists with margin under 2% of the pushes {np.mean(r < 0.02):.2f}, under 1% {np.mean(r < 0.01):.2f}")
    print(f"         on the lead over the runner-up: median sum of |pushes| {np.median(dsum):.1f}; margin / that: median {np.median(rd):.3f}, "
          f"under 2% on {np.mean(rd < 0.02):.2f} of lists, under 10% on {np.mean(rd < 0.1):.2f}")
