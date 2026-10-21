"""v23 with the nine-in-eight leak of ARC's timing put into its readout: position p reads sum_q K(p, q) x (v23's
reading of position q), K = I - n n^T over positions 0-8, n from nine_in_eight.py. Everything else exact. Shows, inside a
working network, how much of ARC's density the forced leak accounts for, by leak shape."""
import numpy as np
from density import measure, T
from nine_in_eight import null_dir
from algzoo.handbuilt_v23 import build_v23
def leaky(net, n):
    import copy
    net = copy.deepcopy(net); wo = net.params[2]
    K = np.eye(10); K[:9, :9] -= np.outer(n, n); wo[:] = K @ wo
    return net
if __name__ == "__main__":
    base = build_v23()
    for name, ang, rho in (("regular 45-degree grid", [0, 45, 90, 135, 180], None), ("ARC's angles", [0, 43, 83, 126, 180], None),
                           ("ARC's angles and sizes", [0, 43, 83, 126, 180], np.array([0.99, 0.94, 0.94, 0.82, 0.90]))):
        measure(f"v23 + leak, {name}", T(leaky(base, null_dir(ang, rho)).params))
