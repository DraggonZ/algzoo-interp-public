"""Do the generic checks work on a different task? Argmedian RNNs (answer = position of the median), 16 and 32 neurons.

1. Accuracy on random inputs.
2. Slide on random inputs: correlation between centered scores after step t+1 (positions 0..n-2) and after step t
   (positions 1..n-1), compared with 2nd-argmax.
3. Weight-level shift relation: output row j versus row j-1 times the step matrix (most common on/off pattern on random
   inputs), with random rows as a baseline.
4. A lone input's leftover score spread versus the typical spread on random inputs.
"""
from collections import Counter

import numpy as np

from algzoo.core import example, load


def analyse(name, params):
    w_hi, w_hh, w_oh = (np.asarray(p, np.float64) for p in params)
    n = w_oh.shape[0]; d = len(w_hi)
    rng = np.random.default_rng(0)
    X = rng.standard_normal((20000, n))
    h = np.zeros((len(X), d)); S = []; pats = Counter()
    for t in range(n):
        pre = h @ w_hh.T + X[:, t, None] * w_hi; h = np.maximum(0, pre)
        s = h @ w_oh.T; S.append(s - s.mean(1, keepdims=True))
        if t >= n // 2:
            for row in (pre[:500] > 0).astype(int):
                pats[tuple(row)] += 1
    S = np.array(S)
    tgt = np.argsort(X, 1)[:, n // 2] if "median" in name else np.argsort(X, 1)[:, -2]
    acc = np.mean(S[-1].argmax(1) == tgt)
    cs = [np.corrcoef(S[t + 1, i, :-1], S[t, i, 1:])[0, 1] for t in range(n // 2, n - 1) for i in range(0, 2000, 7)]
    M = np.array(pats.most_common(1)[0][0], float)[:, None] * w_hh
    Wc = w_oh - w_oh.mean(0)
    rel = np.median([np.dot(Wc[j], Wc[j - 1] @ M) / (np.linalg.norm(Wc[j]) * np.linalg.norm(Wc[j - 1] @ M) + 1e-12) for j in range(1, n)])
    R = rng.standard_normal(Wc.shape); R -= R.mean(0)
    relr = np.median([np.dot(R[j], R[j - 1] @ M) / (np.linalg.norm(R[j]) * np.linalg.norm(R[j - 1] @ M) + 1e-12) for j in range(1, n)])
    hh = np.maximum(0, w_hi); lone = []
    for k in range(1, n):
        hh = np.maximum(0, w_hh @ hh); lone.append(np.ptp(Wc @ hh))
    print(f"{name}: accuracy {acc:.3f}; slide correlation median {np.median(cs):.2f}; shift relation {rel:.2f} (random rows {relr:+.2f}); "
          f"lone-input spread {np.median(lone):.1f} vs random-input spread {np.median(np.ptp(S[-1], 1)):.1f}")


analyse("2nd-argmax 16x10 (ARC's example)", example())
for d in (16, 32):
    for s in (0, 1, 2):
        analyse(f"argmedian {d}x11 seed {s}", load(f"argmedian_{d}_11_1073741824_{s}.pth"))
