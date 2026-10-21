"""Which weights cause each copy's two-spike loss? Restore one weight group to the base values, recompute exactly.

Groups (clean neurons C = 1, 2, 4, 6, 7, 8; dense = the other 10): W_hi; W_oh; W_hh[C, C]; W_hh[C, dense];
W_hh[dense, C]; W_hh[dense, dense]; the five sensitive weights W_hh[7,2], [2,2], [8,7], [1,7], [8,2].
Output: reports/z2_attrib.npz with exact Z_2 and carry loss for each copy and each restored group.
"""
import os
from multiprocessing import Pool

import numpy as np

from algzoo.bench import load_bench
from algzoo.core import example
from z2_stages import stages

C = [1, 2, 4, 6, 7, 8]
Dn = [i for i in range(16) if i not in C]
FIVE = [(7, 2), (2, 2), (8, 7), (1, 7), (8, 2)]
GROUPS = ["none", "W_hi", "W_oh", "hh_CC", "hh_C<-dense", "hh_dense<-C", "hh_dense_dense", "five"]
base = example()


def restore(params, g):
    w_hi, w_hh, w_oh = (p.copy() for p in params)
    if g == "W_hi":
        w_hi = base[0].copy()
    elif g == "W_oh":
        w_oh = base[2].copy()
    elif g == "hh_CC":
        w_hh[np.ix_(C, C)] = base[1][np.ix_(C, C)]
    elif g == "hh_C<-dense":
        w_hh[np.ix_(C, Dn)] = base[1][np.ix_(C, Dn)]
    elif g == "hh_dense<-C":
        w_hh[np.ix_(Dn, C)] = base[1][np.ix_(Dn, C)]
    elif g == "hh_dense_dense":
        w_hh[np.ix_(Dn, Dn)] = base[1][np.ix_(Dn, Dn)]
    elif g == "five":
        for i, j in FIVE:
            w_hh[i, j] = base[1][i, j]
    return w_hi, w_hh, w_oh


def job(args):
    i, params = args
    out = []
    for g in GROUPS:
        o, _, _ = stages(restore(params, g))
        out.append((o["right_T_right_end"] + o["wrong_T_right_end"], o["right_T_wrong_end"]))
    return i, out


if __name__ == "__main__":
    os.environ["OMP_NUM_THREADS"] = "1"
    names, acc, models = load_bench(("mult",))
    with Pool(4) as pool:
        res = dict(pool.imap_unordered(job, list(enumerate(models))))
    R = np.array([res[i] for i in range(len(models))])      # [96, groups, 2]
    np.savez("/data/output/research/algzoo/20261016_run2/reports/z2_attrib.npz", names=names, acc=acc, R=R, groups=np.array(GROUPS))
    sig = np.array([float(str(n).split("_")[0][4:]) for n in names])
    zb = 0.977427
    for s in (0.005, 0.01, 0.02):
        k = sig == s
        loss = zb - R[k, 0, 0]
        print(f"σ={s}: mean Z2 loss {loss.mean():.4f}; share of loss removed by restoring each group:")
        for j, g in enumerate(GROUPS[1:], 1):
            print(f"   {g:16s} {((R[k, j, 0] - R[k, 0, 0]).sum() / loss.sum()):+.2f}   carry loss {R[k, 0, 1].mean():.4f} -> {R[k, j, 1].mean():.4f}")
