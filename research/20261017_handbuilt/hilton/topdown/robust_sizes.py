"""v19 (16 neurons, hides more than the jump: no structural tie), v20 (22) and v21 (28), each with ARC's net vote
sizes per position at a tie (readout rows scaled; v18_recency.K_NET). Same scores as v18_margin.py."""
import numpy as np, torch
import v18_margin as vm
from algzoo.handbuilt_v19 import build_v19
from algzoo.handbuilt_v20 import build_v20
from algzoo.handbuilt_v21 import build_v21
from v18_recency import K_NET
def P(x): return x.params if hasattr(x, "params") else x
if __name__ == "__main__":
    import sys
    base = {"v19 (16, kappa 0.5)": P(build_v19(kappa=0.5)), **({} if "--v19" in sys.argv else {"v20 (22)": P(build_v20()), "v21 (28)": P(build_v21())})}
    V = {}
    for nm, (wi, wh, wo) in base.items():
        V[nm] = (wi, wh, wo); V[nm + " + net sizes"] = (wi, wh, wo * K_NET[:, None])
    vm.__dict__["VARIANTS_OVERRIDE"] = V
    exec(open("v18_margin.py").read().split('if __name__ == "__main__":')[1].replace("variants = {", "variants = VARIANTS_OVERRIDE; _unused = {", 1).replace("\n    ", "\n"), vm.__dict__)
