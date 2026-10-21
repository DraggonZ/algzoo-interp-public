"""Check v23 against its intended scores: relative score alpha x + beta relu(x) everywhere, the current largest at
A0 (p - kappa d), plus lobes of revealed records. First: lists with no overtaking (one record), where the intended scores
are exact; then accuracy on random lists for several loop-back weights."""
import numpy as np, sys
from algzoo.handbuilt_v23 import build_v23, scores
rng = np.random.default_rng(0)
if __name__ == "__main__":
    for w in (0.0, 0.5, 1.0):
        net = build_v23(w=w); A0, al, be = 1.0, 0.5, 0.5
        X = rng.standard_normal((200000, 10)); sc = scores(net, X)
        # intended for lists whose largest is the only record after the first positive... use all lists: compare non-largest positions
        target = al * X + be * np.maximum(X, 0)
        r = X.argmax(1); n = np.arange(len(X))
        rec = X > np.maximum.accumulate(np.concatenate([np.zeros((len(X), 1)), X[:, :-1]], 1), 1)
        nrec = rec.sum(1)  # number of records (first positive counts)
        one = nrec == 1
        err = sc - target; err[n, r] = 0
        print(f"w {w}: lists with one record {one.mean():.1%}; max |score - intended| at other positions {np.abs(err[one]).max():.2e}")
        pm = np.where(one, 0.0, np.nan)
        hid = sc[n, r] - (A0 * X[n, r] - A0 * 1.5 * X[n, r])  # one record: p = 0, d = x_r
        print(f"   largest's position, one record: max |score - A0 (p - kappa d)| {np.abs(hid[one]).max():.2e}")
        acc = (sc.argmax(1) == np.argsort(-X, 1)[:, 1]).mean()
        print(f"   accuracy {acc:.4f}")
