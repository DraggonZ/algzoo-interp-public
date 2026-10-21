"""Review 4 check: length-5 zoo models with the maintained default quadrature (centroid rule on narrow pieces).
table_exact_len5.log was made at 19:00 with the earlier code (commit d097df5: degree 3 below 0.1 rad). Monte Carlo values
are copied from that log (float64, 2^32 samples each)."""
import os
from multiprocessing import Pool

os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np  # noqa: E402

MC = {"2nd_argmax_3_5": 0.37544686, "2nd_argmax_4_5": 0.42845158, "2nd_argmax_5_5": 0.54242799, "2nd_argmax_6_5": 0.70285729,
      "argmedian_8_5": 0.69854302}
OLD = {"2nd_argmax_3_5": 0.37544201, "2nd_argmax_4_5": 0.42845946, "2nd_argmax_5_5": 0.54243797, "2nd_argmax_6_5": 0.70285726,
       "argmedian_8_5": 0.69855112}


def run(stem):
    import algzoo.exact_nd as E
    from algzoo.core import load
    p = load(f"{stem}_1073741824_bo5.pth")
    rank = 3 if "argmedian" in stem else 2
    a, st = E.exact_accuracy(p, rank=rank)
    return stem, a, st["ops"]


if __name__ == "__main__":
    with Pool(5, maxtasksperchild=1) as P:
        for stem, a, ops in P.imap_unordered(run, list(MC)):
            mc = MC[stem]
            se = np.sqrt(mc * (1 - mc) / 2**32)
            print(f"{stem}: default {a:.8f} z {(a - mc) / se:+.2f} | earlier code {OLD[stem]:.8f} z {(OLD[stem] - mc) / se:+.2f} | ops {ops:.3g}", flush=True)
