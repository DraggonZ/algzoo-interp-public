"""Fading memory with the level added uniformly at each jump (ARC's store feeding its level mode, which shrinks 0.99 per
step), and ARC's imprint ratio (1.7 votes per unit of the largest, from steer/lead_parts: memory -83 against +50).
Nudge battery, leak shape and density. Prediction (written first): at rho 0.99 the largest leaks a few hundredths per
unit into near ties, the battery gap falls (near ties softer), maps rise to the hundreds; accuracy stays near 95%."""
import sys; sys.path.insert(0, "../v23")
import numpy as np
from battery import battery, compare, hb_groups, ARC, ARC_GROUPS, T
from algzoo.handbuilt_v23 import build_v23
if __name__ == "__main__":
    nets = {"ARC": (ARC, ARC_GROUPS)}
    for name, kw in (("v23 U1.7", dict(U_mem=1.7)), ("fade .99", dict(U_mem=1.7, rho=0.99, level_exact=False)),
                     ("fade .97", dict(U_mem=1.7, rho=0.97, level_exact=False))):
        net = build_v23(**kw); nets[name] = (T(net.params), hb_groups(net))
    res = {n: battery(W, G) for n, (W, G) in nets.items()}; compare(res)
    import leak_shape, density
    for name in list(nets)[1:]:
        leak_shape.shape(nets[name][0], name); density.measure(name, nets[name][0])
