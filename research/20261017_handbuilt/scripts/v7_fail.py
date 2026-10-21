"""Where v7 (ring-up clock) goes wrong: answer kind on each edge case, and two example lists."""
import numpy as np
from handbuilt_v7 import build_v7, scores
X = np.random.default_rng(31).standard_normal((200_000, 10)); B, n = X.shape
z = np.zeros((B, 2)); m1p = np.stack([np.max(np.c_[z, X[:, :t]], 1) for t in range(n)], 1)
rec = X > m1p; order = X.argsort(1); sec, lar, thi = order[:, -2], order[:, -1], order[:, -3]
cons = (rec[:, :-1] & rec[:, 1:]).any(1); onepos = (X > 0).sum(1) <= 1
np.set_printoptions(precision=1, suppress=True, linewidth=160)
for ck in ("ringup", "late"):
    P = scores(build_v7(clock=ck), X).argmax(1)
    print(f"clock={ck}")
    for name, m in (("record at 9", rec[:, 9]), ("record at 8", rec[:, 8] & ~rec[:, 9]), ("back to back only", cons & ~rec[:, 8] & ~rec[:, 9])):
        m = m & ~onepos; bad = m & (P != sec)
        kinds = {"largest": (P == lar)[bad].mean(), "third": (P == thi)[bad].mean(), "pos 8": (P == 8)[bad].mean(), "pos 9": (P == 9)[bad].mean()}
        print(f"  {name:18s} wrong {bad.sum() / m.sum():.2f};  of wrong: " + ", ".join(f"{k} {v:.2f}" for k, v in kinds.items()))
