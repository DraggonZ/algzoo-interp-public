"""Mechanism checks with a background: the other eight inputs set to b (default -0.5) instead of 0.

Lone input: scores with one input of +1 among the background, minus scores of the background alone (its leftover vote).
Two inputs: largest 1 and second r among the background; vote placed when the later arrives and held; slide on the
steps after both (here the background keeps arriving, so 'slide' means the best slot moves down one).
Tiny model: predicted = background-only scores + m1 * leftover[age] + c * m2 at the second's slot.
"""
import sys

import numpy as np

from algzoo.core import example, zoo

b = float(sys.argv[1]) if len(sys.argv) > 1 else -0.5


def scores(params, x):
    w_hi, w_hh, w_oh = params
    h = np.zeros(len(w_hi)); S = []
    for t in range(10):
        h = np.maximum(0, w_hh @ h + x[t] * w_hi)
        s = w_oh @ h; S.append(s - s.mean())
    return np.array(S)


for name in ["example", 0, 1, 2, 3, 4]:
    params = tuple(np.asarray(p, np.float64) for p in (example() if name == "example" else zoo(16, 10, name)))
    bg = np.full(10, b)
    S0 = scores(params, bg)
    lead = []
    for p in range(10):
        x = bg.copy(); x[p] = 1.0
        d = scores(params, x)[9] - S0[9]
        lead.append(np.sort(d)[-1] - np.sort(d)[-2])
    placed = held = slide = slide_n = n = 0
    for r in (0.3, 0.7, 0.95):
        for p1 in range(10):
            for p2 in range(10):
                if p1 == p2:
                    continue
                x = bg.copy(); x[p1] = 1.0; x[p2] = r
                S = scores(params, x)
                t0 = max(p1, p2); n += 1
                placed += S[t0].argmax() == p2 + (9 - t0)
                held += all(S[t].argmax() == p2 + (9 - t) for t in range(t0, 10))
                for t in range(t0, 9):
                    slide += S[t].argmax() == S[t + 1].argmax() + 1; slide_n += 1
    print(f"{name}: background {b}: lone input's leftover lead median {np.median(lead):.1f}; vote placed {placed}/{n}, held {held}/{n}; slide {slide/slide_n:.1%}")
