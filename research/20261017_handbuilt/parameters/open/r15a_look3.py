"""r15a look 3: the merged rules under the v2c explorer (check against the orchestrator's +197.9 etc.), and the
centred leftover of the current best per readout row and column."""
import numpy as np
import r15a_xc as X

np.set_printoptions(precision=2, suppress=True, linewidth=200)
for f in ["rules/r12a_regpush.py", "rules/r13b_gauge.py", "rules/r14a_minarrival.py"]:
    R = X.load(f)
    net, d = X.price(R)
    print(f, "net %+.1f gain %s" % (net, d["C"]))
wf = d["wf"]
P = wf[272:].reshape(10, 16); A = X.WO
Pc = P - P.mean(0); Ac = A - A.mean(0)
E = Pc - Ac
print("centred leftover rows 0-9\n", E)
print("row rel err (centred)", np.linalg.norm(E, axis=1) / np.linalg.norm(Ac, axis=1))
print("row rest-bit spreads:", {r: round(v[0], 3) for r, v in d["rowerr"].items() if str(r).startswith("out")})
