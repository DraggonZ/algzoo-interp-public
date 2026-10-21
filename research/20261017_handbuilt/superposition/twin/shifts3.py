"""Current twin constants (imprint 1 vote, keep share 0.25, fading 0.99 with uniform level) plus ARC-scale hiding gain
(g 30, 10; detector and store gains 10 as ARC's), for nudge shifts, rates and accuracy."""
from shifts import shifts, ARC, ARC_GROUPS, hb_groups, T
import numpy as np
from battery import battery, compare
from algzoo.handbuilt_v23 import build_v23
BASE = dict(U_mem=1.0, share=0.25, rho=0.99, level_exact=False)
if __name__ == "__main__":
    nets = {"ARC": (ARC, ARC_GROUPS)}
    for name, kw in (("twin", BASE), ("twin g30", dict(BASE, g=30.0)), ("twin g10 a10", dict(BASE, g=10.0, a=10.0, s=10.0))):
        net = build_v23(**kw); nets[name] = (T(net.params), hb_groups(net))
    res = {k: shifts(W, G) for k, (W, G) in nets.items()}
    for k in list(res)[1:]:
        a = np.array([res["ARC"][t] for t in res["ARC"]]); b = np.array([res[k][t] for t in res["ARC"]])
        print(f"{k}: shifts same sign {np.mean(np.sign(a) == np.sign(b)):.0%}; median ratio {np.median(np.abs(b) / np.abs(a)):.2f}; per test " + " ".join(f"{x:.1f}" for x in np.abs(b) / np.abs(a)))
    rb = {k: battery(W, G) for k, (W, G) in nets.items()}; compare(rb)
