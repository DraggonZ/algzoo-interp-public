"""w5b analysis read: in w4b_margin, which neurons must be silenced (one step after the largest arrives) for the network
to name the largest? Uses the frozen version-2 hiding test's own function, with a map whose hiding rows hold the chosen
neurons. Sets: w4b's hiding slots (C1, C6, C7, Q); the four hiding-wave helpers (flip, three clocks); both; the wave plus
C7 (the comparators with the largest: ARC's neuron 7's reference); and others for attribution."""
import sys
import numpy as np
sys.path.insert(0, "/data/output/research/algzoo/20261017_handbuilt/faithful_twin")
from score_wide import Net  # noqa: E402
from score_mechanism2 import names_largest_when_silenced  # noqa: E402
from algzoo.handbuilt_w4b import build_w4b

wi, wh, wo, idx = build_w4b(late=True, level=True, refs=True, cancel=True, margin=True)
n = len(wi); wave = ["flip", "clock0", "clock1", "clock2"]
sets = {"w4b hiding slots: c1 c6 c7 Q": ["c1", "c6", "c7", "Q"],
        "wave only": wave,
        "wave + c7": wave + ["c7"],
        "wave + c7 + c1 + c6": wave + ["c7", "c1", "c6"],
        "wave + c7 + c1 + c6 + Q": wave + ["c7", "c1", "c6", "Q"],
        "wave + c1 + c6 + Q": wave + ["c1", "c6", "Q"]}
for name, ks in sets.items():
    M = np.zeros((16, n)); rows = [1, 6, 7, 8]
    for j, k in enumerate(ks): M[rows[j % 4], idx[k]] = 1.0
    other = [k for k in range(n) if M[:, k].sum() == 0]
    free = [r for r in range(16) if r not in rows]
    for r, k in zip(free, other): M[r, k] = 1.0
    c = names_largest_when_silenced(Net([wi, wh, wo], M))
    print(f"{name:32s} names the largest {100 * c:5.1f}%", flush=True)
