"""Development check on held-out failures: does giving more neurons look-ahead fix seeds 101 and 102?"""
import sys, time, numpy as np
from algzoo.core import load
from prop4 import Prop
G = int(sys.argv[1]); tr = np.load("/data/output/research/algzoo/20261017_handbuilt/hilton/propagate/across.npy")[:, 0]
ALL = list(range(100, 116)) + list(range(200, 216))
for s in (101, 102):
    W = [np.asarray(w, float) for w in load(f"/data/output/research/algzoo/20261016_run3/models/2nd_argmax_16_10_1073741824_{s}.pth")]
    for k in (3, 4, 5):
        t0 = time.time(); P = Prop(W, G=G, lookahead="auto", k_la=k, chunk=1_000_000); a = P.run()
        print(f"seed {s}, look-ahead {k} neurons {P.la}: predicted {100*a:.2f}% vs true {100*tr[ALL.index(s)+1]:.2f}%; misplaced " + " ".join(f"{j}:{d:.2f}" for j, d in sorted(enumerate(P.disc_all), key=lambda q: -q[1])[:7]) + f"  {time.time()-t0:.0f}s", flush=True)
