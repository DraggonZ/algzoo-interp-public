import numpy as np
from algzoo.bench import load_bench
from algzoo.core import example

names, acc, models = load_bench(("mult",))
get = lambda n: example() if n == "example" else models[list(names).index(n)]
np.set_printoptions(linewidth=200, precision=3, suppress=True)
for pat in ["1111..111.111111", "1111..111..11111", "111...111.1.1111"]:
    D = np.array([c == "1" for c in pat])
    print("pattern", pat)
    for nm in ["example", "mult0.02_18", "mult0.02_3", "mult0.01_14", "mult0.005_3"]:
        w_hh = get(nm)[1]
        A = w_hh[np.ix_(D, D)]
        ev = np.linalg.eigvals(A)
        ev = ev[np.argsort(-np.abs(ev))]
        print(f"  {nm:12s} |λ| {np.abs(ev)[:10]}  angle/2π {np.angle(ev[:10]) / 2 / np.pi}")
