"""Test: is the forgetting ratio (twin_forget.py, nudge 0.01) a small-change effect? Replace a
number at position 3 or 5 by a fresh value from the same class, so it stays forgotten (uniform in (0, 0.75 x second-
largest before it)) or stays kept (uniform in [0.75 x second, largest before it)), excluding lists where it is in the
top two. Score change per unit of value change, median; ratio forgotten / kept. Trained network, twin, v7."""
import io, contextlib
import numpy as np
with contextlib.redirect_stdout(io.StringIO()):
    import twin_forget as tf
import simlib

X = tf.X; rng = np.random.default_rng(4)
for name, (P, _) in tf.nets.items():
    L = simlib.run(X, params=P)[0]; rat = []
    for s_ in (3, 5):
        B = len(X); Pp = np.c_[np.zeros((B, 2)), X[:, :s_]]; m2 = np.sort(Pp, 1)[:, -2]; m1 = Pp.max(1); x = X[:, s_]
        top = (np.argsort(X, 1)[:, -2:] == s_).any(1)
        forg = (x > 0) & (x < 0.75 * m2) & ~top; kept = (x > 0) & (x >= 0.75 * m2) & ~top & (x < m1)
        new = np.where(forg, rng.uniform(0, 1, B) * 0.75 * m2, np.where(kept, 0.75 * m2 + rng.uniform(0, 1, B) * (m1 - 0.75 * m2), x))
        Xn = X.copy(); Xn[:, s_] = new; dx = np.abs(new - x)
        ok = dx > 1e-3
        gs = np.linalg.norm(simlib.run(Xn, params=P)[0] - L, axis=1) / np.maximum(dx, 1e-3)
        rat.append((np.median(gs[forg & ok]) / np.median(gs[kept & ok]), np.median(dx[forg & ok]), np.median(dx[kept & ok])))
    print(f"{name:26s}: redraw ratio forgotten/kept: position 3 {rat[0][0]:.3f}, position 5 {rat[1][0]:.3f} (median change of the number: forgotten {rat[0][1]:.2f}, kept {rat[0][2]:.2f})", flush=True)
