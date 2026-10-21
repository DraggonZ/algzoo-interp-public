"""How alike are the matched weights? Cosine similarity between the trained network's weights and the matched
hand-built ones, block by block (rows = to, columns = from). Groups: largest pair (2, 4), gates (5, 9),
hiding (1, 6, 7, 8), memory (0, 3, 10-15). Large hand-built vetoes (|w| > 100) are clipped to 100 first, since one
such weight would decide a cosine alone."""
import sys
import numpy as np
from algzoo.core import example
T = [np.asarray(p, float) for p in example()]
G = {"largest": [2, 4], "gates": [5, 9], "hiding": [1, 6, 7, 8], "memory": [0, 3, 10, 11, 12, 13, 14, 15]}
def cos(a, b):
    a, b = a.ravel(), np.clip(b.ravel(), -100, 100); return a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)
for f in sys.argv[1:]:
    d = np.load(f); H = (d["wi"], d["wh"], d["wo"])
    print(f"== {f}")
    print(f"   input weights: {cos(T[0], H[0]):.2f}")
    for gt, rt in G.items():
        print(f"   recurrent into {gt:8s}", "  ".join(f"from {gf}: {cos(T[1][np.ix_(rt, rf)], H[1][np.ix_(rt, rf)]):5.2f}" for gf, rf in G.items()))
    print("   readout       ", "  ".join(f"{gf}: {cos(T[2][:, rf], H[2][:, rf]):5.2f}" for gf, rf in G.items()))
    rng = np.random.default_rng(0); null = [cos(T[1], H[1][np.ix_(p, p)]) for p in (rng.permutation(16) for _ in range(200))]
    print(f"   whole recurrent matrix {cos(T[1], H[1]):.2f}  (random relabelling: median {np.median(null):.2f}, 95th pct {np.percentile(null, 95):.2f})")
