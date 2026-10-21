"""Review 4 check: exact accuracy of the eight length-4 zoo models under three quadrature settings of exact_nd.py.

default = maintained code now (centroid rule on pieces narrower than 0.2 rad, degree 7 otherwise)
old     = code at table_exact_len4.log time (commit d097df5: degree 3 below 0.1 rad, degree 7 otherwise)
deg7    = degree 7 everywhere (review 3's converged setting)
Monte Carlo references: every logged run, pooled by sample count (float32 and float64 shown separately for M5,4).
"""
import os
import sys
from multiprocessing import Pool

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
import numpy as np  # noqa: E402

D = [2, 3, 4, 5, 6, 8, 16, 32]
SETTINGS = {"default": {}, "old": {"NARROW": 0.1, "NARROW_S": 1}, "deg7": {"GM_S": 3}}


def run(args):
    d, s = args
    import algzoo.exact_nd as E
    from algzoo.core import load
    E.NARROW, E.NARROW_S, E.GM_S = 0.2, 0, None          # reset to the maintained defaults (workers are reused)
    for k, v in SETTINGS[s].items():
        setattr(E, k, v)
    acc, st = E.exact_accuracy(load(f"2nd_argmax_{d}_4_1073741824_bo5.pth"))
    return d, s, acc, st["ops"]


# (value, n) for every logged Monte Carlo run
MC = {2: [(0.32802305, 2**32), (0.32802763, 2**32)], 3: [(0.43328767, 2**32)], 4: [(0.60201549, 2**32), (0.60202938, 2**32)],
      5: [(0.74567132, 2**32), (0.74563564, 2**32), (0.74564823, 2**32),                          # float32
          (0.74565318, 2**32), (0.74564473, 2**32), (0.74565415, 2**32), (0.74567229, 2**31)],     # float64
      6: [(0.86475940, 2**32)], 8: [(0.98745941, 2**32), (0.98745991, 2**32)], 16: [(0.99830649, 2**32), (0.99830735, 2**32)],
      32: [(0.99879956, 2**32), (0.99880068, 2**32)]}


def pool(runs):
    n = sum(m for _, m in runs)
    p = sum(v * m for v, m in runs) / n
    return p, np.sqrt(p * (1 - p) / n)


if __name__ == "__main__":
    jobs = [(d, s) for d in D for s in SETTINGS]
    with Pool(7, maxtasksperchild=1) as P:
        res = P.map(run, jobs, chunksize=1)
    R = {(d, s): (a, o) for d, s, a, o in res}
    for d in D:
        p, se = pool(MC[d])
        line = f"M{d},4  MC pooled {p:.8f} ± {se:.1e} ({len(MC[d])} runs)"
        for s in SETTINGS:
            a, o = R[(d, s)]
            line += f" | {s} {a:.8f} z {(a - p) / se:+.2f}"
        print(line)
    for lab, runs in [("float32 only", MC[5][:3]), ("float64 only (4 runs)", MC[5][3:]), ("float64, 3 runs of mc_m5_4.log", MC[5][3:6]),
                      ("review 3's two runs", [MC[5][0], MC[5][6]])]:
        p, se = pool(runs)
        print(f"M5,4 {lab}: {p:.8f} ± {se:.1e}; z of deg7 {(R[(5, 'deg7')][0] - p) / se:+.2f}, of default {(R[(5, 'default')][0] - p) / se:+.2f}")
    f32 = np.array([v for v, _ in MC[5][:3]])
    chi2 = ((f32 - f32.mean()) ** 2).sum() / (MC[5][0][0] * (1 - MC[5][0][0]) / 2**32)
    print(f"M5,4 float32 scatter: chi2 {chi2:.1f} on 2 dof")
    sys.stdout.flush()
