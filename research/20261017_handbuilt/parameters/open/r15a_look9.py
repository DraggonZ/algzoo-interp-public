"""r15a look 9: the rule r15a_contender.py and r14a in both explorers, list seed from FIT_SEED."""
import os
import r12a_x as X2
import r15a_xc as XC
print("FIT_SEED", os.environ.get("FIT_SEED", "101"))
for f in ["rules/r14a_minarrival.py", "rules/r15a_contender.py"]:
    R = X2.load(f)
    a = X2.price(R)[0]; b, d = XC.price(R)
    print(f"{f:32s} v2 {a:+.1f}  v2c {b:+.1f}  gain {d['C'][0]:.3f}  struct {d['struct']}", flush=True)
