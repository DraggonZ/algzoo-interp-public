"""How often do the decoded votes at step t name the second-largest so far (the running answer)?
Evaluation only; nothing is fitted."""
import numpy as np
from decoders import *
from step2d_age_decoder import patterns_by_age, P_seq
D = patterns_by_age(np.random.default_rng(11).standard_normal((100000, 10)))
on, _ = common_pattern(run(np.random.default_rng(3).standard_normal((100000, 10))))
X = np.random.default_rng(5).standard_normal((30000, 10)); H = run(X)
print(" t | running answer named among positions seen: common decoder / age decoder / zero continuation")
for t in range(2, 11):
    run2 = np.argsort(X[:, :t], 1)[:, -2]
    vb = H[:, t] @ pushback_full(t, on).T
    age = (t - 1) - X[:, :t].argmax(1)
    vc = np.zeros_like(vb)
    for a in np.unique(age):
        m = age == a; vc[m] = H[m, t] @ (WOC if t == 10 else P_seq(D[(t, a)])).T
    vz = zero_split(H[:, t], t).sum(1)
    f = lambda v: (v[:, :t].argmax(1) == run2).mean()
    print(f"{t:2d} | {f(vb):.3f} / {f(vc):.3f} / {f(vz):.3f}")
