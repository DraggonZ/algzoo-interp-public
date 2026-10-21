import numpy as np
from handbuilt_v8 import build_v7, scores
X = np.random.default_rng(30).standard_normal((300_000, 10)); B, n = X.shape
z = np.zeros((B, 2)); m1p = np.stack([np.max(np.c_[z, X[:, :t]], 1) for t in range(n)], 1)
rec = X > m1p; sec = X.argsort(1)[:, -2]
cons = (rec[:, :-1] & rec[:, 1:]).any(1)
onepos = (X > 0).sum(1) <= 1
for name, kw in (("v7 + z, 17 neurons", dict(clock="z")), ("v8 = v7 + z - second store, 16 neurons", dict(clock="z", second=False))):
    net = build_v7(**kw); P = scores(net, X).argmax(1); ok = P == sec
    print(f"{name} ({len(net.params[0])}): all {ok.mean():.3f}")
    for name, m in (("new largest at step 9", rec[:, 9]), ("new largest at step 8", rec[:, 8]), ("back-to-back new largests", cons),
                    ("none of these, 2+ positives", ~rec[:, 9] & ~rec[:, 8] & ~cons & ~onepos), ("at most one positive", onepos)):
        print(f"   {name:30s} share {m.mean():.2f}  accuracy {ok[m].mean():.3f}")
