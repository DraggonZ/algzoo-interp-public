"""Figure 2's two tests on full random lists.
(1) Sufficiency: silence neurons 1, 6, 7, 8 by hand at a step t after the largest arrives (no bigger number exists).
    Prediction: the largest's hidden timer becomes a vote, so the answer moves to the largest's position.
    Controls: no intervention; four random other neurons silenced (10 sets).
(2) Necessity: lists whose second-largest arrives before the largest. At the largest's step, restore the four to their
    values in the same list with the largest replaced by -0.46 (no bigger number). Prediction: the network loses the
    answer. Controls: no intervention; four random other neurons restored (10 sets)."""
import numpy as np

from algzoo.core import example

wi, wh, wo = example(); LOW = [1, 6, 7, 8]; OTHER = [k for k in range(16) if k not in LOW]
rng = np.random.default_rng(0)


def run(X, t_set=None, G=None, V=None):
    h = np.zeros((len(X), 16)); hs = []
    for t in range(10):
        h = np.maximum(0, h @ wh.T + X[:, t:t + 1] * wi)
        if t_set is not None:
            m = t_set == t
            h[np.ix_(m, G)] = 0 if V is None else V[m, t][:, G]
        hs.append(h)
    return (h @ wo.T).argmax(1), np.stack(hs, 1)


X = rng.standard_normal((200_000, 10)); o = np.argsort(-X, 1); p1, p2 = o[:, 0], o[:, 1]
# (1) sufficiency: largest positive, arrives at p1 <= 7; silence at a random step in (p1, 9]
s = (X[np.arange(len(X)), p1] > 0) & (p1 <= 8)
Xs, q1 = X[s], p1[s]; t = np.array([rng.integers(a + 1, 10) for a in q1])
base, _ = run(Xs)
ans, _ = run(Xs, t, LOW)
ctrl = np.mean([np.mean(run(Xs, t, list(rng.choice(OTHER, 4, replace=False)))[0] == q1) for _ in range(10)])
print(f"(1) {len(Xs)} random lists; answer = the largest's position: no intervention {100*np.mean(base == q1):.1f}%, "
      f"silence 1, 6, 7, 8 {100*np.mean(ans == q1):.1f}%, silence four random others {100*ctrl:.1f}%")
for lo, hi in ((1, 2), (2, 5), (5, 10)):
    m = (t - q1 >= lo) & (t - q1 < hi)
    print(f"    silenced {lo}-{hi-1} steps after the largest: {100*np.mean(ans[m] == q1[m]):.1f}% (n={m.sum()})")
# (2) necessity: second before largest
s2 = p2 < p1
Xn, n1, n2 = X[s2], p1[s2], p2[s2]
Xc = Xn.copy(); Xc[np.arange(len(Xn)), n1] = -0.46
_, Hc = run(Xc)
base2, _ = run(Xn)
ans2, _ = run(Xn, n1, LOW, Hc)
ctrl2 = np.mean([np.mean(run(Xn, n1, list(rng.choice(OTHER, 4, replace=False)), Hc)[0] == n2) for _ in range(10)])
print(f"(2) {len(Xn)} random lists with the second before the largest; network right: no intervention {100*np.mean(base2 == n2):.1f}%, "
      f"restore 1, 6, 7, 8 {100*np.mean(ans2 == n2):.1f}%, restore four random others {100*ctrl2:.1f}%")
