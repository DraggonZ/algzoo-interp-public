"""Where v17 (or any builder kwargs) loses lists: by situation of the second-largest."""
import numpy as np, sys, json
sys.path.insert(0, "scripts")
from handbuilt_v17 import build_v17, scores
X = np.random.default_rng(30).standard_normal((200_000, 10)); B = len(X)
m1p = np.stack([np.max(np.c_[np.zeros((B, 2)), X[:, :t]], 1) for t in range(10)], 1); rec = X > m1p
o = X.argsort(1); sec, lar, thi = o[:, -2], o[:, -1], o[:, -3]
i = np.arange(B); secrec = rec[i, sec]
def breakdown(kw):
    P = scores(build_v17(**kw), X).argmax(1); bad = P != sec
    groups = {
        "second = old largest, current right after": secrec & (lar == sec + 1),
        "second = old largest, current 2+ after": secrec & (lar >= sec + 2),
        "second kept, after the largest": ~secrec & (sec > lar),
        "second kept, before the largest": ~secrec & (sec < lar),
    }
    print(kw, f"accuracy {1 - bad.mean():.3f}")
    for k, m in groups.items():
        lost = (bad & m).mean()
        print(f"  {k:45s} share {m.mean():.3f}  right {1 - bad[m].mean():.3f}  lost {lost:.3f}" +
              f"  | wrong names: largest {(P == lar)[bad & m].mean():.2f} third {(P == thi)[bad & m].mean():.2f} pos8-9 {(P >= 8)[bad & m].mean():.2f}")
if __name__ == "__main__":
    breakdown(json.loads(sys.argv[1]) if len(sys.argv) > 1 else dict(gamma=0.25, fwd=True, dl=True))
