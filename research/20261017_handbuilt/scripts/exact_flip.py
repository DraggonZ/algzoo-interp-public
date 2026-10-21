"""Pilot: find close-call flip points exactly instead of by 24-step bisection.
The network is piecewise linear in the inputs. Along the line x(s) = base + s * e_k (the third number's value s),
one forward pass with a tangent gives every pre-activation z and its slope dz/ds, and the score margin m and dm/ds.
Inside the current on/off pattern, m is exactly linear: the root is s - m/(dm/ds). If no pre-activation changes sign
before the root, the root is exact; otherwise step just past the first sign change and repeat.
Compares with bisection on ARC's network for the 720 placements x 7 sizes of the largest (second-third flips)."""
import itertools

import numpy as np

from algzoo.core import example

wi, wh, wo = example()
pl = np.array(list(itertools.permutations(range(10), 3))); N = len(pl); r = np.arange(N)
GRID = [1.05, 1.15, 1.3, 1.6, 2.0, 3.0, 5.0]


def margin(X, a, b):
    h = np.zeros((len(X), 16))
    for t in range(10):
        h = np.maximum(0, h @ wh.T + X[:, t:t + 1] * wi)
    s = h @ wo.T
    return s[np.arange(len(X)), b] - s[np.arange(len(X)), a]


def bisect(base, k, a, b, lo, hi, steps=24):
    lo, hi = np.full(len(base), lo), hi.copy()
    for _ in range(steps):
        mid = (lo + hi) / 2; X = base.copy(); X[np.arange(len(X)), k] = mid
        f = margin(X, a, b) > 0
        hi = np.where(f, mid, hi); lo = np.where(f, lo, mid)
    return (lo + hi) / 2


def exact(base, k, a, b, s0, lo, hi, max_iter=20):
    """Piecewise-linear root finding. Returns flip value and the number of passes used."""
    n = len(base); s = s0.copy(); done = np.zeros(n, bool); passes = np.zeros(n, int); out = np.full(n, np.nan)
    dwo = wo[b] - wo[a]
    for _ in range(max_iter):
        X = base.copy(); X[np.arange(n), k] = s
        h = np.zeros((n, 16)); dh = np.zeros((n, 16)); zs, dzs = [], []
        for t in range(10):
            z = h @ wh.T + X[:, t:t + 1] * wi
            dz = dh @ wh.T + (k == t)[:, None] * wi
            on = z > 0
            h, dh = np.where(on, z, 0), np.where(on, dz, 0)
            zs.append(z); dzs.append(dz)
        m = (dwo * h).sum(1); dm = (dwo * dh).sum(1)
        step = np.where(dm != 0, -m / np.where(dm != 0, dm, 1), np.inf)
        Z, DZ = np.stack(zs, 1), np.stack(dzs, 1)            # (n, 10, 16)
        with np.errstate(divide="ignore", invalid="ignore"):
            cross = -Z / DZ                                  # step at which each pre-activation hits zero
        same_dir = np.sign(cross) == np.sign(step)[:, None, None]
        cross = np.where(same_dir & (np.abs(cross) > 1e-12), np.abs(cross), np.inf).reshape(n, -1).min(1)
        passes += ~done
        fin = ~done & (np.abs(step) <= cross)
        out[fin] = (s + step)[fin]; done |= fin
        if done.all():
            break
        mv = ~done
        s[mv] = s[mv] + np.sign(step[mv]) * (cross[mv] + 1e-9)
        s = np.clip(s, lo, hi)
    return out, passes


diffs, passes_all, nan = [], [], 0
for g in GRID:
    base = np.full((N, 10), -0.46); base[r, pl[:, 0]] = g; base[r, pl[:, 1]] = 1.0
    hi = np.full(N, min(1.7, g - 1e-3))
    bb = bisect(base, pl[:, 2], pl[:, 1], pl[:, 2], 0.3, hi)
    ex, ps = exact(base, pl[:, 2], pl[:, 1], pl[:, 2], np.full(N, 1.0), 0.3, hi)
    ok = ~np.isnan(ex); nan += (~ok).sum()
    diffs.append(np.abs(ex[ok] - bb[ok])); passes_all.append(ps)
    print(f"largest {g}: |exact - bisection| median {np.median(np.abs(ex[ok]-bb[ok])):.1e}, max {np.max(np.abs(ex[ok]-bb[ok])):.1e}; "
          f"passes mean {ps.mean():.2f}, max {ps.max()}; unresolved {(~ok).sum()}")
P = np.concatenate(passes_all)
print(f"overall: passes per flip point mean {P.mean():.2f} (bisection uses 24); share needing one pass {np.mean(P == 1):.2f}; "
      f"unresolved {nan} of {len(P)}; bias differences above 1e-6: {np.mean(np.concatenate(diffs) > 1e-6):.3f}")
