"""r3b, variant 3: r3b_qx relabelled (the same function): the runner-up comparator Q in ARC's slot 8, the third clock
neuron in slot 14. Rules: algzoo.handbuilt_v27b. Constants read off ARC: 4 (memory imprint 1, keep share 0.25, memory
fade 0.99, keep cut 0.75 of the runner-up) + 16 loudness scales."""
import numpy as np
from algzoo.handbuilt_v27b import build_v27b
from algzoo.handbuilt_v26 import to_arc


def weights():
    wi, wh, wo = to_arc(build_v27b(level_exact=True), fixed={3: "mem6"}, free=(1, 2, 5, 7))
    # to_arc puts the third clock in 8 and Q (by correlation) in 14; swap them (rescaled to ARC's partner's RMS by
    # to_arc already for the old slots; here the swap keeps each neuron's own scale)
    P = np.arange(16); P[8], P[14] = 14, 8
    return wi[P], wh[np.ix_(P, P)], wo[:, P]
