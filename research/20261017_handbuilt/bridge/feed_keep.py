"""Direct test: does silencing the hiding neurons stop the memory from receiving an overtaken record's wave?
For lists with a record at r overtaken at q, run a counterfactual copy where the overtaking number is replaced by a
number just below the record (x_r - 0.01), so the hiding neurons are not silenced. In the real run, from step q on,
the memory receives the hiding neurons' feed from the counterfactual copy instead of its own (wh[memory, hiding] times
the counterfactual hiding values); everything else is the real run. Then the double nudge of tail_cancel2.py (record
and overtaking number moved together; the counterfactual's replacement moves with the record). Prediction if the
hypothesis holds: the old record's wave reappears at positions after q."""
import numpy as np
from algzoo.core import example
wi, wh, wo = (np.asarray(p, float) for p in example())
MEM = [0, 3, 10, 11, 12, 13, 14, 15]; HID = [1, 6, 7, 8]
rng = np.random.default_rng(3); X = rng.standard_normal((1500000, 10))
def run(X, r, q, keep):
    Xc = X.copy(); Xc[:, q] = X[:, r] - 0.01
    h = np.zeros((len(X), 16)); hc = np.zeros((len(X), 16))
    for t in range(10):
        prc = hc @ wh.T + Xc[:, t:t + 1] * wi
        pre = h @ wh.T + X[:, t:t + 1] * wi
        if keep and t > q:     # the feed computed from the state after step q onward
            pre[:, MEM] += (hc[:, HID] - h[:, HID]) @ wh[np.ix_(MEM, HID)].T
        h, hc = np.maximum(pre, 0), np.maximum(prc, 0)
    return h
eps = 1e-3
for r, q in [(1, 4), (2, 5), (2, 7), (4, 7)]:
    pre = np.maximum(X[:, :r].max(1), 0)
    ok = (X[:, r] > pre + 0.05) & (X[:, r] > 0.3) & (X[:, r + 1:q].max(1) < X[:, r]) & (X[:, q] > X[:, r] + 0.05)
    if q < 9: ok &= X[:, q + 1:].max(1) < X[:, q]
    Xs = X[ok]; Xp = Xs.copy(); Xp[:, r] += eps; Xp[:, q] += eps
    for keep in (False, True):
        tot = np.median((run(Xp, r, q, keep) - run(Xs, r, q, keep)) @ wo.T / eps, 0)
        print(f"r{r} q{q} {'feed kept' if keep else 'as is    '}:", " ".join(f"{v:7.1f}" for v in tot), f"| after q sum {tot[q + 1:9].sum():7.1f}")
