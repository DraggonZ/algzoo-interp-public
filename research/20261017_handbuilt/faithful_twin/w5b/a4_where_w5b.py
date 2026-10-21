"""w5b analysis read: in w5b (argument: copy_share), what does the network name when the hiding slots (four wave neurons, C1, C6, Q) are
silenced one step after the largest arrives, and how do the scores of the largest and the second change? 200,000 lists
with the largest at steps 0-8. Scores are taken relative to the second-largest's score."""
import numpy as np
from algzoo.handbuilt_w5b import build_w5b

wi, wh, wo, idx = build_w5b(copy_share=float(__import__("sys").argv[1]))
rng = np.random.default_rng(5); X = rng.standard_normal((400_000, 10)); L = X.argmax(1); X, L = X[L <= 8], L[L <= 8]
S2 = np.argsort(-X, 1)[:, 1]; r = np.arange(len(X))
def run(ks):
    h = np.zeros((len(X), len(wi)))
    for t in range(10):
        h = np.maximum(h @ wh.T + X[:, t:t + 1] * wi, 0)
        m = L + 1 == t
        for k in ks: h[m, idx[k]] = 0
    return h @ wo.T
for name, ks in (("intact", []), ("hiding slots silenced", ["flip", "clock0", "clock1", "clock2", "c1", "c6", "Q"])):
    sc = run(ks); ans = sc.argmax(1)
    rel = sc - sc[r, S2][:, None]
    print(f"{name}: names largest {100 * (ans == L).mean():.1f}%, second {100 * (ans == S2).mean():.1f}%, other {100 * ((ans != L) & (ans != S2)).mean():.1f}%")
    d = X[r, L] - X[r, S2]
    print(f"   largest's score minus second's: median {np.median(rel[r, L]):.3f}; per unit of (largest - second): median {np.median(rel[r, L] / d):.3f}")
    off = (np.arange(10)[None] - L[:, None])
    for k in (1, 2, 3, -1, -2):
        ok = (L + k >= 0) & (L + k <= 9)
        print(f"   position largest{k:+d}: score minus second's, median {np.median(rel[r[ok], (L + k)[ok]]):.3f}; named {100 * (ans == L + k).mean():.1f}%")
