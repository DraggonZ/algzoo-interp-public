"""Step 4b (seed 41, fits seed 32): (i) neuron 3 with one shared per-age value for the largest and the second
(12 numbers instead of 23); (ii) for compact-65, which neuron's rule causes the loss on lists with a new largest at
step 9 and on lists whose last new largest came at steps 0-2 (points recovered by putting it back on its own sign)."""
import numpy as np
from sklearn.linear_model import LogisticRegression
from swlib import *
import final as fl

Xf = lists(FIT_SEED, 200_000); lr3, w15 = fl.fit(Xf); col = fl.col; oh = fl.oh
_, Pf, _ = run(Xf); agf, _ = order_keys(Xf)
D = (oh(agf[..., 0]) + oh(agf[..., 1]))[:, 1:].reshape(-1, 11)
lrs = LogisticRegression(C=10.0, max_iter=2000).fit(D, (Pf[:, 1:, 3] <= 0).ravel())
print("neuron 3 shared per-age value (ages -1..9): " + " ".join(f"{v:.1f}" for v in lrs.coef_[0]) + f"  constant {lrs.intercept_[0]:.2f}")
X = lists(41, 200_000); K = kinds(X); S0, P0, H0 = run(X); base = score(X, S0, S0, K); true_off = P0[:, :, MEM] <= 0
ag, _ = order_keys(X)
R = fl.rules(X, lr3, w15, "compact-65")
R2 = R.copy(); R2[:, 1:, col[3]] = (lrs.decision_function((oh(ag[..., 0]) + oh(ag[..., 1])).reshape(-1, 11)).reshape(len(X), 10) > 0)[:, 1:]
for name, RR in (("compact-65", R), ("compact-65 with 3 on one shared per-age value (54 numbers)", R2)):
    S, offs = machine(X, lambda t, c: RR[:, t]); report(name, X, S0, S, offs, true_off, base)
y = X.argsort(1)[:, -2]; V = task(X); last = np.array([np.max(np.nonzero(r)[0]) if r.any() else -1 for r in V["rec"]])
groups = {"new largest at step 9": last == 9, "last new largest at steps 0-2": (last >= 0) & (last <= 2), "rest": (last >= 3) & (last <= 8)}
ok0 = S0.argmax(1) == y; S, _ = machine(X, lambda t, c: R[:, t]); ok = S.argmax(1) == y
print("compact-65 change by group: " + " ".join(f"{g}: {100 * (ok[m].mean() - ok0[m].mean()):+.2f}" for g, m in groups.items()))
for n in MEM:
    Sn, _ = machine(X, lambda t, c: R[:, t], use=[m for m in MEM if m != n]); okn = Sn.argmax(1) == y
    print(f"   {n} back on its own sign, recovered: " + " ".join(f"{g}: {100 * (okn[m].mean() - ok[m].mean()):+.2f}" for g, m in groups.items()))
