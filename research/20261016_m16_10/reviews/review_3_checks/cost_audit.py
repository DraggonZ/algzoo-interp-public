"""Review 3 check: which operations does exact_nd's multiply-add count leave out on M16,4?

Wraps `cut` to count, per call, the adjacency test the counter skips: the integer product
(~T)[k, m] @ common[m, P] (k*m*P multiply-adds) plus forming `common` (P*m). Wraps qhull's Delaunay to count
calls, points and wall time. Same algorithm, same answer; only extra counters.
Usage: OMP_NUM_THREADS=1 python cost_audit.py [model-name]
"""
import sys
import time

import numpy as np

import algzoo.exact_nd as E
from algzoo.core import load

orig_cut = E.cut
extra = {"adj": 0, "common": 0, "cuts": 0}
dl = {"calls": 0, "points": 0, "time": 0.0}
RealDelaunay = E.Delaunay


def cut(R, H, b, ops):
    k, n = R.shape
    s = R @ (b / np.linalg.norm(b))
    pos, neg = s > E.TOL, s < -E.TOL
    extra["cuts"] += 1
    if pos.any() and neg.any():
        T = np.abs(R @ H.T) < 1e-11
        ip, jn = np.flatnonzero(pos), np.flatnonzero(neg)
        P = len(ip) * len(jn)
        ii, jj = np.repeat(ip, len(jn)), np.tile(jn, len(ip))
        ok = (T[ii] & T[jj]).sum(1) >= n - 2
        extra["common"] += P * H.shape[0]
        extra["adj"] += k * H.shape[0] * int(ok.sum())
    return orig_cut(R, H, b, ops)


def delaunay(Y, **kw):
    t0 = time.time()
    out = RealDelaunay(Y, **kw)
    dl["time"] += time.time() - t0
    dl["calls"] += 1
    dl["points"] += len(Y)
    return out


E.cut = cut
E.Delaunay = delaunay
name = sys.argv[1] if len(sys.argv) > 1 else "2nd_argmax_16_4_1073741824_bo5.pth"
params = load(name)
d, n = params[1].shape[0], params[2].shape[0]
per_sample = n * (d * d + d) + n * d
t0 = time.time()
acc, st = E.exact_accuracy(params)
wall = time.time() - t0
print(f"{name}: exact {acc:.8f}, regions {st['regions']}, wall {wall:.1f}s (instrumented)")
print(f"  counted multiply-adds {st['ops']:.3g} = {st['ops'] / per_sample:,.0f} samples (enumeration {st['ops_enum'] / per_sample:,.0f})")
print(f"  uncounted adjacency product {extra['adj']:.3g} = {extra['adj'] / per_sample:,.0f} samples; "
      f"pair-constraint ANDs {extra['common']:.3g} = {extra['common'] / per_sample:,.0f} samples; cuts {extra['cuts']:,}")
print(f"  qhull Delaunay: {dl['calls']:,} calls, {dl['points']:,} points, {dl['time']:.1f}s of {wall:.1f}s wall")
print(f"  counted + adjacency = {(st['ops'] + extra['adj']) / per_sample:,.0f} samples")
