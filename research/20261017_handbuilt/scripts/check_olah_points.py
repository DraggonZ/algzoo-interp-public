"""Check two points about the step-0 swap test and neuron 5.
(1) Are neurons 1, 6, 7, 8 zero at the wipe step in both lists (making the step-0 swap test automatic)? Also on random lists.
(2) Does neuron 5 fire one step after every new largest, and does switching it off then lose that number's later vote?"""
import itertools

import numpy as np

from algzoo.core import example

wi, wh, wo = example()
L = [1, 6, 7, 8]


def run(x, off=None):
    h = np.zeros(16); hs = []
    for t, v in enumerate(x):
        h = np.maximum(0, wh @ h + wi * v)
        if off is not None and off[0] == t:
            h = h.copy(); h[off[1]] = 0
        hs.append(h)
    return wo @ h, hs


zero = []; n5 = []; broke = []; ok_before = []
for a, p1 in itertools.permutations(range(10), 2):
    if a >= p1:
        continue
    x = np.zeros(10); x[a] = 1.0; x[p1] = 1.5
    s, hs = run(x)
    zero.append(np.all(hs[p1][L] == 0)); n5.append(hs[a + 1][5] > 0 if a + 1 < 10 else np.nan)
    ok_before.append(s.argmax() == a)
    if a + 1 < p1:
        s2, _ = run(x, (a + 1, 5)); broke.append(s2.argmax() != a)
print(f"two-number lists: four are zero at the wipe step in {np.mean(zero):.2f}; right before intervention {np.mean(ok_before):.2f}")
print(f"neuron 5 fires one step after the smaller number (then the largest so far): {np.nanmean(n5):.2f}")
print(f"switching neuron 5 off at that one step makes the answer wrong in {np.mean(broke):.2f} of {len(broke)} lists")
rng = np.random.default_rng(0); X = rng.standard_normal((20000, 10))
z = []
for x in X:
    o = np.argsort(-x); p1, a = o[0], o[1]
    if a < p1:
        _, hs = run(x); z.append(np.mean(hs[p1][L] == 0))
print(f"random lists, second before largest: share of the four at zero at the wipe step: mean {np.mean(z):.3f}, all four zero in {np.mean(np.array(z) == 1):.3f}")
# Lone numbers: does a new largest's timer depend on neuron 5? Timer = projection on the twelve neurons later.
