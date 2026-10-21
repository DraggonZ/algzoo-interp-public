"""Analysis read: why does reading position 0 from the runner-up's value fail? Compare each position's score with
A0 x (the vote the twin's docstring promises) on 20,000 lists."""
import numpy as np
import algzoo.handbuilt_v23 as v23
from algzoo.handbuilt_v27b import build_v27b

rng = np.random.default_rng(1); X = rng.standard_normal((20_000, 10)); o = np.argsort(-X, 1); y = o[:, 1]
import sys
net = build_v27b(level_exact=bool(int(sys.argv[1]))); sc = v23.scores(net, X)
print("level_exact", sys.argv[1])
s9 = np.maximum(X[np.arange(len(X)), y], 0)
print("score_0 - 0.99 s_9: median abs", np.median(np.abs(sc[:, 0] - 0.99 * s9)))
sec = sc[np.arange(len(X)), y]
print("second's score - its value (lists with second not at 0): median", np.median((sec - X[np.arange(len(X)), y])[y != 0]),
      " 10/90%", np.percentile((sec - X[np.arange(len(X)), y])[y != 0], [10, 90]))
pred = sc.argmax(1); wrong = pred != y
print("accuracy", 1 - wrong.mean(), "; wrong answers naming position 0:", (wrong & (pred == 0)).mean(),
      "; second at 0 but missed:", (wrong & (y == 0)).mean())
# shared offset: score minus value for every non-largest position
lg = o[:, 0]
d = sc - X; mask = np.ones_like(sc, bool); mask[np.arange(len(X)), lg] = False; mask[:, 0] = False
print("score - value, non-largest positions 1-9, by position (median):", np.round([np.median(d[:, p][mask[:, p]]) for p in range(1, 10)], 3))
print("relu part: score - (0.75 x + 0.25 relu(x)) median by position:",
      np.round([np.median((sc[:, p] - 0.75 * X[:, p] - 0.25 * np.maximum(X[:, p], 0))[mask[:, p]]) for p in range(1, 10)], 3))
w0 = wrong & (pred == 0); v2 = X[np.arange(len(X)), y]
print("wrong-0 lists: second <= 0:", (w0 & (v2 <= 0)).mean(), "; second > 0:", (w0 & (v2 > 0)).mean())
m = w0 & (v2 > 0)
print("  of these, the second was an overtaken largest (largest after it):", (m & (lg > y)).mean(),
      "; second's score / value median:", np.median(sec[m] / v2[m]))
print("  position of the second in these lists:", np.bincount(y[m], minlength=10), "; position of the largest:", np.bincount(lg[m], minlength=10))
mm = wrong & (y == 0)
print("second at 0 missed: named a position before the largest:", (mm & (pred < lg)).mean(), "; after:",
      (mm & (pred > lg)).mean(), "; the largest:", (mm & (pred == lg)).mean())
