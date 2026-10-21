"""Is zeroing neurons 1, 6, 7, 8 by itself enough to unhide a timer, and is it needed?
Sufficiency: a lone 1 at position a, zeros elsewhere. At step t > a set the four to zero by hand (no bigger number).
Does the answer become a? Control: zero 4 random other neurons instead.
Necessity: lists with 1.0 at a and 1.5 at p1 > a. At step p1 replace the four by their values in the run without the
1.5 (an un-wipe). Is the answer still a?"""
import itertools

import numpy as np

from algzoo.core import example

wi, wh, wo = example()
L = [1, 6, 7, 8]; others = [k for k in range(16) if k not in L]
rng = np.random.default_rng(0)


def run(x, t_set=None, G=None, vals=None):
    h = np.zeros(16); hs = []
    for t, v in enumerate(x):
        h = np.maximum(0, wh @ h + wi * v)
        if t_set == t:
            h = h.copy(); h[G] = 0 if vals is None else vals[G]
        hs.append(h)
    s = wo @ h
    return s - s.mean(), hs


suff, ctrl, base_votes, wipe_votes = [], [], [], []
for a, t in itertools.permutations(range(10), 2):
    if t <= a:
        continue
    x = np.zeros(10); x[a] = 1.0
    s0, _ = run(x); s, _ = run(x, t, L)
    suff.append(s.argmax() == a); base_votes.append(s0[a]); wipe_votes.append(s[a] - np.delete(s, a).max())
    for _ in range(10):
        s, _ = run(x, t, list(rng.choice(others, 4, replace=False))); ctrl.append(s.argmax() == a)
print(f"lone 1, four zeroed by hand one or more steps later: answer = the 1's position in {np.mean(suff):.2f} of {len(suff)}; "
      f"zeroing 4 random other neurons: {np.mean(ctrl):.2f}")
print(f"  score of the 1's position without the wipe: mean {np.mean(base_votes):.1f}; with it, margin over the next best: median {np.median(wipe_votes):.1f}")
nec, nec_ctrl = [], []
for a, p1 in itertools.permutations(range(10), 2):
    if p1 <= a:
        continue
    x = np.zeros(10); x[a] = 1.0; x[p1] = 1.5
    xn = x.copy(); xn[p1] = 0
    _, hn = run(xn)
    s, _ = run(x, p1, L, hn[p1]); nec.append(s.argmax() == a)
    G = list(rng.choice(others, 4, replace=False)); s, _ = run(x, p1, G, hn[p1]); nec_ctrl.append(s.argmax() == a)
print(f"two-number lists, four un-wiped at the bigger number's step: answer still right in {np.mean(nec):.2f} of {len(nec)}; "
      f"un-wiping 4 random other neurons instead: {np.mean(nec_ctrl):.2f}")
