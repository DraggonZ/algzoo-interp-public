import numpy as np
from handbuilt_v8 import build_v7, scores
X = np.random.default_rng(30).standard_normal((200_000, 10)); B = len(X)
m1p = np.stack([np.max(np.c_[np.zeros((B, 2)), X[:, :t]], 1) for t in range(10)], 1); rec = X > m1p
o = X.argsort(1); sec, lar, thi = o[:, -2], o[:, -1], o[:, -3]
cons = (rec[:, :-1] & rec[:, 1:]).any(1); onepos = (X > 0).sum(1) <= 1
ordinary = ~rec[:, 8] & ~rec[:, 9] & ~cons & ~onepos
for kw in (dict(clock="z"), dict(clock="z", second=False)):
    P = scores(build_v7(**kw), X).argmax(1); bad = ordinary & (P != sec)
    secrec = rec[np.arange(B), sec]
    print(kw, f"wrong {bad.sum() / ordinary.sum():.3f}; of wrong: names largest {(P == lar)[bad].mean():.2f}, third {(P == thi)[bad].mean():.2f};"
          f" second was a record {secrec[bad].mean():.2f} (all ordinary {secrec[ordinary].mean():.2f}); named position 8 {(P == 8)[bad].mean():.2f}, 9 {(P == 9)[bad].mean():.2f};"
          f" third negative {(X[np.arange(B), thi] < 0)[bad].mean():.2f}")
    print("   wrong rate by position of the second:", [round(((P != sec) & ordinary & (sec == p)).sum() / max(1, (ordinary & (sec == p)).sum()), 2) for p in range(10)])
