"""Which realisation change moves v23 toward ARC on the nudge battery? Diagnostic: each variant's weights come from
v23's rules plus one change (a gain or angle read off ARC, or ARC's forced leak); nothing is tuned to the battery."""
import sys; sys.path.insert(0, "../v23")
import numpy as np
from battery import battery, compare, hb_groups, ARC, ARC_GROUPS, T
from algzoo.handbuilt_v23 import build_v23
from leak_net import leaky
from nine_in_eight import null_dir
if __name__ == "__main__":
    base = build_v23()
    nets = {"ARC": (ARC, ARC_GROUPS)}
    for name, net in (("v23", base), ("gain 10", build_v23(a=10.0, s=10.0, g=10.0)), ("loop 0.25", build_v23(w=0.25)),
                      ("ARC leak", leaky(base, null_dir([0, 43, 83, 126, 180], np.array([0.99, 0.94, 0.94, 0.82, 0.90]))))):
        nets[name] = (T(net.params), hb_groups(net))
    res = {n: battery(W, G) for n, (W, G) in nets.items()}
    compare(res); np.save("variants.npy", res, allow_pickle=True)
