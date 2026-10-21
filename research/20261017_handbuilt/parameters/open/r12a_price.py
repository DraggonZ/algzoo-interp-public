"""Price a rule with the explorer (exact v2 net without KL). Usage: python r12a_price.py rules/<rule>.py"""
import sys
import r12a_x as X

R = X.load(sys.argv[1])
print("STRUCTURE_BITS", R.STRUCTURE_BITS)
net, d = X.price(R, True)
for r, (e, s) in sorted(d["rowerr"].items()):
    print(f"{r:6s} err {e:.3f} size {s:.3f} rel {e / s:.3f}")
