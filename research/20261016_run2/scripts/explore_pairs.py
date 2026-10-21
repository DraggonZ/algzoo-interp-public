import numpy as np
from algzoo.bench import load_bench
from algzoo.core import example
from algzoo.exact_z2 import exact_z2

names, acc, models = load_bench(("mult",))
z = np.load("../reports/z2_check.npz")
zz = dict(zip(z["names"], z["z2_exact"]))
np.set_printoptions(linewidth=200, precision=2, suppress=True)
for label, params in [("example", example()), ("mult0.02_18", models[list(names).index("mult0.02_18")]),
                      ("mult0.02_3", models[list(names).index("mult0.02_3")]), ("mult0.01_14", models[list(names).index("mult0.01_14")])]:
    tot, ops, nc, det = exact_z2(params, return_detail=True)
    M = np.full((10, 10), np.nan)
    for (p1, p2), (a, good) in det.items():
        M[p1, p2] = a
    print(f"\n{label}: Z_2 {tot:.4f}. per-pair accuracy, rows p1 (max position), cols p2 (2nd position)")
    print(M)
