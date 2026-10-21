"""The shuffled-error control on the twin (v18): for each account of all 16 neurons (twin_graded.py's rules: exact,
conceptual misreadings, right idea with one constant off), substitute (a) the account's values and (b) the true values
plus the account's own errors shuffled across lists. Same size and spread at every neuron; only the structure differs.
Gains as twin_graded.py (default 1e4, or args g w a). 50,000 lists."""
import os
import sys
import numpy as np
os.environ.pop("CURVE", None); os.environ.pop("NEAR", None); os.environ.pop("SAVE", None)
argv = sys.argv[:]; sys.argv = sys.argv[:1] + argv[1:4]
import io, contextlib
with contextlib.redirect_stdout(io.StringIO()):
    import twin_graded as tg  # runs its default grid once on import (output suppressed); its rules and memory are reused
from simlib import lists, run

X = lists(1919, 50_000); y = X.argsort(1)[:, -2]; P = tg.P
tg.X = X; V = tg.task_vars(X); tg.x, tg.m1, tg.jump, tg.age = V["x"], V["m1"], V["jump"], V["age"]; tg.V = V
prev_rec, _, _ = tg.record_vars(X)
tg.d_true = np.where(V["age"] >= 0, V["m1in"] - np.where(V["newmax"] > 0, V["m1"], prev_rec), 0); tg.d_true = np.where(V["newmax"] > 0, 0, tg.d_true)
L, _, H = run(X, params=P); rng = np.random.default_rng(1)
print(f"twin gains {tg.g_:g}: accuracy {(L.argmax(1) == y).mean():.3f}")
cases = [("conceptual", "wrong clock height", {"height": "largest"}), ("conceptual", "flip term left out", {"flip": False}),
         ("one constant off", "clock angle 45.5", {"deg": 45.5}), ("one constant off", "clock angle 46", {"deg": 46.0}),
         ("one constant off", "gate cutoff 0.765", {"c": 0.765}), ("one constant off", "store gain 2% off", {"sg": 20.4})]
for kind, label, kw in cases:
    E = tg.rules(**kw); M = tg.memory(E)
    sims = {n: (M[:, :, tg.MEM.index(n)] if n in tg.MEM else E[n]) for n in range(16)}
    perm = rng.permutation(len(X))
    shuf = {n: np.maximum(H[:, :, n] + (sims[n] - H[:, :, n])[perm], 0) for n in range(16)}
    a = (run(X, sims, params=P)[0].argmax(1) == y).mean(); s = (run(X, shuf, params=P)[0].argmax(1) == y).mean()
    print(f"{kind:17s} {label:20s}: accuracy {a:.3f}, with its errors shuffled {s:.3f}", flush=True)
