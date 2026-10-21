"""Fixed lists for the top-down substitution tests: every ordered position of the three largest numbers, their values
on a grid (third u, gaps a and b, ../propagate/full_mean.py grid 16 x 16 x 16), the other seven at the mean of a normal
truncated below the third. Saves X (lists) and w (exact probability weight of each list, summing to 1)."""
import sys, itertools
import numpy as np
from scipy.stats import norm
sys.path.insert(0, "../propagate")
V, wt = __import__("full_mean").grid(16, 16)
T = np.array(list(itertools.permutations(range(10), 3)))
bg = -norm.pdf(V[:, 2]) / norm.cdf(V[:, 2])
X = np.repeat(bg[None, :, None], len(T), 0).repeat(10, 2)            # [720, nV, 10]
for k in range(3):
    X[np.arange(len(T))[:, None], np.arange(len(V))[None, :], T[:, k][:, None]] = V[None, :, k]
X = X.reshape(-1, 10); w = np.tile(wt, len(T)); w = w / w.sum()
np.savez("grid_lists.npz", X=X, w=w); print(X.shape, w.sum())
