"""Held-out check: the frozen estimator (settings fixed on ARC: G, automatic look-ahead of 3 neurons, tie spacing 1/3 of
a cell, cells by the top-three density, last number exact) applied unchanged to the 32 other trained networks of run 3.
Truth: ../../../hilton/propagate/across.npy column 0 (10M lists each; rows ARC, seeds 100-115, 200-215)."""
import sys, time, numpy as np
from algzoo.core import load
from prop4 import Prop
G = int(sys.argv[1]); ALL = list(range(100, 116)) + list(range(200, 216)); seeds = [int(a) for a in sys.argv[2:]] or ALL
tr = np.load("/data/output/research/algzoo/20261017_handbuilt/hilton/propagate/across.npy")[:, 0]
rows = []
for s in seeds:
    i = ALL.index(s)
    W = [np.asarray(w, float) for w in load(f"/data/output/research/algzoo/20261016_run3/models/2nd_argmax_16_10_1073741824_{s}.pth")]
    t0 = time.time(); P = Prop(W, G=G, lookahead="auto", chunk=2_000_000); a = P.run(); t = tr[i + 1]
    rows.append((s, t, a, P.share_L)); print(f"seed {s}: true {100*t:.2f}%  predicted {100*a:.2f}%  error {100*(a-t):+.2f}  look-ahead {P.la}  {time.time()-t0:.0f}s", flush=True)
e = np.array([r[2] - r[1] for r in rows]) * 100
print(f"{len(rows)} networks: RMSE {np.sqrt((e**2).mean()):.2f} points, mean error {e.mean():+.2f}, largest |error| {np.abs(e).max():.2f}; correlation {np.corrcoef([r[1] for r in rows], [r[2] for r in rows])[0,1]:.3f}")
np.save(f"across_G{G}.npy", np.array(rows))
