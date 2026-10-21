"""Anatomy of close-call margins at an exact tie (idealized-network step 1).
At second = third = 1.0, largest g at p1, the other seven at -0.46, the margin score_p3 - score_p2 equals
sum_t c_t x_t exactly (no biases; c = gradient inside the current on/off pattern). Split it into:
  votes   = c_p3 * 1 + c_p2 * 1     (the two equal numbers' votes not cancelling: unequal vote sizes)
  largest = c_p1 * g                (the largest number's contribution: e.g. its hidden timer leaking)
  backgr  = sum_other c_t * (-0.46) (the seven small numbers)
and report each part's share of the margin's spread across the 720 placements."""
import itertools
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from pl_flip import contributions  # noqa: E402

from algzoo.core import example, zoo  # noqa: E402

pl = np.array(list(itertools.permutations(range(10), 3))); N = len(pl); r = np.arange(N)


def anatomy(P, g=1.6):
    X = np.full((N, 10), -0.46); X[r, pl[:, 0]] = g; X[r, pl[:, 1]] = 1.0; X[r, pl[:, 2]] = 1.0
    parts, c = contributions(P, X, pl[:, 1], pl[:, 2])
    votes = parts[r, pl[:, 1]] + parts[r, pl[:, 2]]; largest = parts[r, pl[:, 0]]
    backgr = parts.sum(1) - votes - largest; m = parts.sum(1)
    return m, votes, largest, backgr, c


if __name__ == "__main__":
    for name, P in (("ARC", example()), ("zoo0", zoo(16, 10, 0)), ("zoo2", zoo(16, 10, 2))):
        m, v, L, B, c = anatomy(P)
        tot = np.var(m)
        share = lambda x: np.cov(x, m)[0, 1] / tot              # additive share of the margin's variance
        c3 = c[r, pl[:, 2]]; c2 = c[r, pl[:, 1]]
        print(f"{name}: margin at tie: median |m| {np.median(np.abs(m)):.2f}; vote slopes: third median {np.median(c3):+.1f}, second median {np.median(c2):+.1f}")
        print(f"   share of margin variance: unequal votes {share(v):+.2f}, largest {share(L):+.2f}, background {share(B):+.2f}")
        print(f"   typical size (median |.|): votes {np.median(np.abs(v)):.2f}, largest {np.median(np.abs(L)):.2f}, background {np.median(np.abs(B)):.2f}; "
              f"third's slope negative in {np.mean(c3 < 0):.2f} of placements")
        for lab, sel in (("second before largest", pl[:, 1] < pl[:, 0]), ("second after largest", pl[:, 1] > pl[:, 0])):
            mm = m[sel]; print(f"   {lab}: n={sel.sum()}, median |m| {np.median(np.abs(mm)):.2f}, shares votes {np.cov(v[sel], mm)[0,1]/np.var(mm):+.2f} largest {np.cov(L[sel], mm)[0,1]/np.var(mm):+.2f} background {np.cov(B[sel], mm)[0,1]/np.var(mm):+.2f}")
