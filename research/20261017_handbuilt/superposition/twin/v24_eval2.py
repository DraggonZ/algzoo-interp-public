"""v24 with every vote fading 0.99 per step since arrival (age_fade), imprint 1.5, last number direct only or gated."""
import numpy as np
from shifts import shifts, ARC, ARC_GROUPS, T
from battery import battery, compare
from v24_eval import groups
from algzoo.handbuilt_v23 import build_v23
from algzoo.handbuilt_v24 import build_v24
if __name__ == "__main__":
    nets = {"ARC": (ARC, ARC_GROUPS)}
    tw = build_v23(U_mem=1.0, share=0.25, rho=0.99, level_exact=False); nets["twin16"] = (T(tw.params), groups(tw))
    for name, kw in (("v24 direct9", dict(age_fade=True, U_mem=1.5)), ("v24 gated9", dict(age_fade=True, U_mem=1.5, last_gate=True))):
        net = build_v24(**kw); nets[name] = (T(net.params), groups(net))
    res = {k: shifts(W, G) for k, (W, G) in nets.items()}
    for k in list(res)[1:]:
        a = np.array([res["ARC"][t] for t in res["ARC"]]); b = np.array([res[k][t] for t in res["ARC"]])
        print(f"{k}: shifts same sign {np.mean(np.sign(a) == np.sign(b)):.0%}; median ratio {np.median(np.abs(b) / np.abs(a)):.2f}; per test " + " ".join(f"{x:.1f}" for x in np.abs(b) / np.abs(a)))
    rb = {k: battery(W, G) for k, (W, G) in nets.items()}; compare(rb)
    import density
    for k in list(nets)[2:]: density.measure(k, nets[k][0])
