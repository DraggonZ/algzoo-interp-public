"""What places the vote? Scores right after the later of the two inputs arrives.

Largest first: does the second's arrival put the top vote on the newest position (9 at that step, i.e. the second)?
Second first: does the largest's arrival put the top vote on the second's position?
Then: does the top vote stay on the sliding target until the end?
"""
import numpy as np

from algzoo.core import example

w_hi, w_hh, w_oh = example()


def scores(x):
    h = np.zeros(16); S = []
    for t in range(10):
        h = np.maximum(0, w_hh @ h + x[t] * w_hi)
        s = w_oh @ h
        S.append(s - s.mean())
    return np.array(S)


for r in (0.3, 0.7, 0.95):
    res = {"max first": [0, 0, 0], "2nd first": [0, 0, 0]}
    for p1 in range(10):
        for p2 in range(10):
            if p1 == p2:
                continue
            x = np.zeros(10); x[p1] = 1.0; x[p2] = r
            S = scores(x)
            t0 = max(p1, p2)
            # at step t0 the second sits at 'slot' p2 + (9 - t0) in sliding coordinates: the vote that will end at p2
            target_now = p2 + (9 - t0)
            lab = "max first" if p1 < p2 else "2nd first"
            res[lab][0] += 1
            res[lab][1] += S[t0].argmax() == target_now
            res[lab][2] += all(S[t].argmax() == p2 + (9 - t) for t in range(t0, 10))
    print(f"ratio {r}: " + "; ".join(f"{k}: vote placed right {v[1]} of {v[0]}, held all the way {v[2]}" for k, v in res.items()))
