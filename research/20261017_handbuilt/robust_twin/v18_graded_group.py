"""The same group-only design on v18 (fragile twin): the timer (clock 4-6, flip 3) substituted by its rules
(gao/twin_graded.py) with the same spoils, everything else as built; and the store 2% too large. Gains 1e4 and 30."""
import sys
import numpy as np
src = open("../gao/twin_graded.py").read(); src = src[:src.index("def memory(E)")]
for gains in ((1e4, 1e4, 1000.0), (30.0, 30.0, 20.0)):
    sys.argv = ["x"] + [str(v) for v in gains]
    g = {}; exec(src, g)
    X, P, L, H, rules, run, CL, F = (g[k] for k in ("X", "P", "L", "H", "rules", "run", "CL", "F"))
    y = X.argsort(1)[:, -2]; T = CL + [F]
    acc = lambda sims: np.mean(run(X, sims, params=P)[0].argmax(1) == y)
    res = {"network": acc({}), "exact timer rules": acc({n: rules()[n] for n in T})}
    for lab, kw in (("misreading: height from the largest", dict(height="largest")), ("misreading: flip left out", dict(flip=False)),
                    ("slightly off: clock at 45.5 degrees", dict(deg=45.5))):
        E = rules(**kw); res[lab] = acc({n: E[n] for n in T})
    res["slightly off: store 2% too large"] = acc({0: 1.02 * H[:, :, 0]})
    print(f"v18 gains {gains[0]:g}: " + "; ".join(f"{k} {v:.3f}" for k, v in res.items()), flush=True)
