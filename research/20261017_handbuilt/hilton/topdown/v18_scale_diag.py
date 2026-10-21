"""Why does scaling v18's readout rows by a few percent break it? On random lists: v18 with row 0 or row 9 scaled by
1.04 alone; accuracy, and the score of the right answer and of the winner in the broken lists."""
import numpy as np
from algzoo.handbuilt_v18 import build_v18
wi, wh, wo = build_v18().params
X = np.random.default_rng(0).standard_normal((200000, 10)); y = np.argsort(X, 1)[:, -2]
def run(wo_):
    h = np.zeros((len(X), 16))
    for t in range(10): h = np.maximum(h @ wh.T + X[:, t:t + 1] * wi, 0)
    return h @ wo_.T
S0 = run(wo); a0 = S0.argmax(1); print(f"v18 {100*(a0==y).mean():.2f}%")
for p in (0, 1, 5, 8, 9):
    w2 = wo.copy(); w2[p] *= 1.04; S = run(w2); a = S.argmax(1); br = (a0 == y) & (a != y)
    print(f"row {p} x1.04: {100*(a==y).mean():.2f}%  broken {100*br.mean():.2f}%  winner pos {np.bincount(a[br], minlength=10)}  "
          f"median score: winner {np.median(S[br, a[br]]) if br.any() else 0:.3g}, right {np.median(S[br, y[br]]) if br.any() else 0:.3g}, "
          f"row {p} terms' size {np.median(np.abs(S0[:, p])):.3g}")
