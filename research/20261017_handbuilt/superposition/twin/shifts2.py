"""v23 with constants read off ARC: memory imprint U = 1 vote per unit of the largest (the slot then holds the number's
shortfall below the largest; also the smallest imprint that keeps slots on), keep-gate share 0.25 (ARC's keep-gate
nudge moves the lead by about a quarter of v23's 0.5 share), fading 0.99 with the level added uniformly."""
from shifts import shifts, ARC, ARC_GROUPS, hb_groups, T
import numpy as np
from battery import battery, compare
from algzoo.handbuilt_v23 import build_v23
if __name__ == "__main__":
    nets = {"ARC": (ARC, ARC_GROUPS)}
    for name, kw in (("U1", dict(U_mem=1.0)), ("U1 keep.25", dict(U_mem=1.0, share=0.25)),
                     ("U1 keep.25 fade", dict(U_mem=1.0, share=0.25, rho=0.99, level_exact=False))):
        net = build_v23(**kw); nets[name] = (T(net.params), hb_groups(net))
    res = {k: shifts(W, G) for k, (W, G) in nets.items()}
    print(f"{'lead shift, in units of the second’s vote (median)':72s}" + "".join(f"{k:>16s}" for k in res))
    for t in res["ARC"]: print(f"{t:72s}" + "".join(f"{res[k][t]:+16.3f}" for k in res))
    for k in list(res)[1:]:
        a = np.array([res["ARC"][t] for t in res["ARC"]]); b = np.array([res[k][t] for t in res["ARC"]])
        print(f"{k}: same sign {np.mean(np.sign(a) == np.sign(b)):.0%}; median ratio of sizes {np.median(np.abs(b) / np.abs(a)):.2f}; per test " + " ".join(f"{x:.1f}" for x in np.abs(b) / np.abs(a)))
    rb = {k: battery(W, G) for k, (W, G) in nets.items()}; compare(rb)
