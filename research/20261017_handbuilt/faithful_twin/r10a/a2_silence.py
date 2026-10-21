"""r10a analysis 2: score_mechanism2's hiding test (silence a set of neurons for one step, the step after the largest;
how often is the largest then named?) on the 18-neuron probe build, for several choices of hiding set.
ARC 81.3%, bar 40.7%. Same construction as the scorer (lists with the largest at steps 0-8), own lists (seed 31, numpy)."""
import numpy as np
from algzoo.handbuilt_v10a import build_v10a

net = build_v10a(ring_tap=5, gap_eff=1.0, last_comp=True)
wi, wh, wo = net.params; ix = net.idx
rng = np.random.default_rng(31); X = rng.standard_normal((400_000, 10)); L = X.argmax(1); k = L <= 8; X, L = X[k], L[k]


def named_largest(sil):
    h = np.zeros((len(X), len(wi)))
    for t in range(10):
        h = np.maximum(h @ wh.T + X[:, t:t + 1] * wi, 0.0)
        m = L + 1 == t
        if sil: h[np.ix_(m, sil)] = 0.0
    return ((h @ wo.T).argmax(1) == L).mean()


C = [ix[f"clock{j}"] for j in range(3)]
sets = {"none": [], "E1, E6, clock2, runner-up (the layout job 7's references need)": [ix["E1"], ix["E6"], C[2], ix["runnerup"]],
        "E1, E6 only": [ix["E1"], ix["E6"]],
        "clock2 only": [C[2]],
        "three clocks + runner-up (r6a's slots)": C + [ix["runnerup"]],
        "three clocks only": C}
for name, s in sets.items():
    print(f"silence {name}: names the largest {100 * named_largest(s):.1f}%")
