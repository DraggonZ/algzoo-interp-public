"""How complex is the near-tie rule from positions and the two ratios (F2)? Accuracy against model size. Ceiling from
the top three alone: redrawing the smaller numbers flips 17% of decisions, so if 2p(1-p) averages 0.17 the best
top-three-only rule is right about 90%."""
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from learnable import pos, rat, y
F = np.c_[pos, rat]
for leaves, it in ((2, 1), (4, 10), (8, 10), (8, 50), (16, 100), (63, 400), (255, 800)):
    clf = HistGradientBoostingClassifier(max_iter=it, learning_rate=0.1 if it > 1 else 1.0, max_leaf_nodes=leaves).fit(F[:300000], y[:300000])
    p = clf.predict_proba(F[300000:])[:, 1]
    print(f"{it:4d} trees x {leaves:3d} leaves (about {it*leaves:6d} numbers): held-out {100*((p > 0.5) == y[300000:]).mean():.1f}%", flush=True)
