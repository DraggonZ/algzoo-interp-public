"""Review 6, check D (pilot for a proposed direction): do the seven small inputs act like zero-mean noise on the deciding margin?

Hypothesis: replacing the seven smallest inputs by their conditional mean preserves accuracy because their real values move
the deciding margin up or down about equally, so mistakes they cause and fix cancel. Seed 1 breaks this (+1.4 points).
For random inputs with the second and third within 0.1 (and the largest more than 0.1 above), compare the margin
(score at the second's slot minus score at the third's slot) on the full and the reduced input. Report the mean and spread
of the change, in score units and as an equivalent shift in value (change / slope, slope = median d margin / d gap fitted
on the reduced inputs), and the mistakes caused and fixed in this subset. Six trained models only; CPU.
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "2")
import numpy as np
import torch as th
from scipy.stats import norm

from algzoo.core import example, zoo

th.set_num_threads(2)
N = 2 ** 22


def scores(params, X):
    w_hi, w_hh, w_oh = (th.as_tensor(np.asarray(q), dtype=th.float32) for q in params)
    out = []
    for xb in th.split(X, 2 ** 19):
        h = th.zeros(len(xb), len(w_hi))
        for t in range(10):
            h = th.relu(h @ w_hh.T + xb[:, t, None] * w_hi)
        out.append(h @ w_oh.T)
    return th.cat(out).numpy()


g = np.random.default_rng(7)
X = g.standard_normal((N, 10)).astype(np.float32)
o = np.argsort(X, 1); ar = np.arange(N)
p1, p2, p3 = o[:, -1], o[:, -2], o[:, -3]
m1, m2, m3 = X[ar, p1], X[ar, p2], X[ar, p3]
sel = (m2 - m3 < 0.1) & (m1 - m2 > 0.1) & (m3 > 0)
cm = (-norm.pdf(m3) / norm.cdf(m3)).astype(np.float32)
small = np.ones_like(X, dtype=bool); small[ar, p1] = small[ar, p2] = small[ar, p3] = False
Xr = np.where(small, cm[:, None], X)
Xs, Xrs = th.as_tensor(X[sel]), th.as_tensor(Xr[sel])
gap = (m2 - m3)[sel]; q2, q3 = p2[sel], p3[sel]; k = np.arange(sel.sum())
print(f"subset: second-third gap < 0.1, largest > 0.1 above, third positive: {100 * sel.mean():.1f}% of inputs")
for name in ["example", 0, 1, 2, 3, 4]:
    p = example() if name == "example" else zoo(16, 10, name)
    S, Sr = scores(p, Xs), scores(p, Xrs)
    M, Mr = S[k, q2] - S[k, q3], Sr[k, q2] - Sr[k, q3]
    rf, rr = S.argmax(1) == q2, Sr.argmax(1) == q2
    # slope of the reduced margin against the gap (least squares through each placement is costly; use pooled fit)
    A = np.vstack([gap, np.ones_like(gap)]).T
    slope = np.linalg.lstsq(A, Mr, rcond=None)[0][0]
    d = M - Mr
    print(f"{name}: margin change mean {d.mean():+.2f} (median {np.median(d):+.2f}), spread {d.std():.2f} score units; slope {slope:.0f} per unit gap -> "
          f"mean shift {d.mean() / slope:+.4f}, spread {d.std() / slope:.4f} in value; in subset: wrong full {100 * (~rf).mean():.1f}%, reduced {100 * (~rr).mean():.1f}%, "
          f"caused {100 * (~rf & rr).mean():.1f}%, fixed {100 * (rf & ~rr).mean():.1f}%", flush=True)
