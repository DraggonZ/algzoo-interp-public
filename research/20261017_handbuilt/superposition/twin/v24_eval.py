"""v24 (17 neurons, runner-up validation) on the nudge battery and lead shifts, beside ARC and the 16-neuron twin. The
runner-up neuron counts as a hiding neuron, as ARC's neuron 8 (hiding and runner-up)."""
import sys
import numpy as np
from shifts import shifts, ARC, ARC_GROUPS, hb_groups, T
from battery import battery, compare
from algzoo.handbuilt_v23 import build_v23
from algzoo.handbuilt_v24 import build_v24
def groups(net):
    G = hb_groups(net)
    if "runner" in net.idx: G["hiding"] = G["hiding"] + [net.idx["runner"]]
    return G
if __name__ == "__main__":
    nets = {"ARC": (ARC, ARC_GROUPS)}
    tw = build_v23(U_mem=1.0, share=0.25, rho=0.99, level_exact=False); nets["twin16"] = (T(tw.params), groups(tw))
    for name, kw in (("v24 U1", dict(last_gate=True)), ("v24 U1.5", dict(U_mem=1.5, last_gate=True))):
        net = build_v24(**kw); nets[name] = (T(net.params), groups(net))
    res = {k: shifts(W, G) for k, (W, G) in nets.items()}
    print(f"{'lead shift, in units of the second’s vote (median)':72s}" + "".join(f"{k:>12s}" for k in res))
    for t in res["ARC"]: print(f"{t:72s}" + "".join(f"{res[k][t]:+12.3f}" for k in res))
    for k in list(res)[1:]:
        a = np.array([res["ARC"][t] for t in res["ARC"]]); b = np.array([res[k][t] for t in res["ARC"]])
        print(f"{k}: shifts same sign {np.mean(np.sign(a) == np.sign(b)):.0%}; median ratio {np.median(np.abs(b) / np.abs(a)):.2f}; per test " + " ".join(f"{x:.1f}" for x in np.abs(b) / np.abs(a)))
    rb = {k: battery(W, G) for k, (W, G) in nets.items()}; compare(rb)
    import density
    for k in list(nets)[1:]: density.measure(k, nets[k][0])
