"""Do the scores literally slide down one position per step?

For two-spike inputs, record the centered scores after every step. On quiet steps after both spikes, compare the scores
at step t+1 for positions 0..8 with the scores at step t for positions 1..9. Also on random full inputs (every step has an
input), to see how much the input itself disturbs the slide.
"""
import numpy as np

from algzoo.core import example

w_hi, w_hh, w_oh = example()
r = 0.7


def scores(x):
    h = np.zeros(16); S = []
    for t in range(10):
        h = np.maximum(0, w_hh @ h + x[t] * w_hi)
        s = w_oh @ h
        S.append(s - s.mean())
    return np.array(S)


cors, tops = [], []
for p1 in range(10):
    for p2 in range(10):
        if p1 == p2:
            continue
        x = np.zeros(10); x[p1] = 1.0; x[p2] = r
        S = scores(x)
        for t in range(max(p1, p2) + 1, 9):
            a, b = S[t + 1][:9], S[t][1:]
            cors.append(np.corrcoef(a, b)[0, 1])
            tops.append(S[t].argmax() == S[t + 1].argmax() + 1)
cors = np.array(cors)
print(f"two-spike, quiet steps ({len(cors)} step pairs): correlation of shifted scores median {np.median(cors):.3f}, 10th percentile {np.percentile(cors,10):.3f}")
print(f"  best position moves down exactly one: {np.mean(tops):.1%}")

rng = np.random.default_rng(0)
X = rng.standard_normal((2000, 10))
cs = []
for x in X:
    S = scores(x)
    for t in range(2, 9):
        cs.append(np.corrcoef(S[t + 1][:9], S[t][1:])[0, 1])
print(f"random inputs: correlation of shifted scores median {np.median(cs):.3f}, 10th percentile {np.percentile(cs,10):.3f}")
x = np.zeros(10); x[2] = 1.0; x[5] = r
print("example: largest at 2, second at 5; centered scores after each step (rows = step):")
np.set_printoptions(linewidth=200, precision=1, suppress=True)
print(scores(x))
