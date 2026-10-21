import numpy as np, torch as th
from algzoo.core import example, forward
X = np.random.default_rng(30).standard_normal((300_000, 10)); B, n = X.shape
z = np.zeros((B, 2)); m1p = np.stack([np.max(np.c_[z, X[:, :t]], 1) for t in range(n)], 1)
rec = X > m1p; sec = X.argsort(1)[:, -2]
cons = (rec[:, :-1] & rec[:, 1:]).any(1); onepos = (X > 0).sum(1) <= 1
ok = forward(example(), th.tensor(X)).numpy().argmax(1) == sec
print(f"ARC all {ok.mean():.3f}")
for name, m in (("new largest at step 9", rec[:, 9]), ("new largest at step 8", rec[:, 8]), ("back-to-back new largests", cons),
                ("none of these, 2+ positives", ~rec[:, 9] & ~rec[:, 8] & ~cons & ~onepos), ("at most one positive", onepos)):
    print(f"   {name:30s} share {m.mean():.2f}  accuracy {ok[m].mean():.3f}")
