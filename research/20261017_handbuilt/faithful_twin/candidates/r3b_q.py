"""r3b, variant 1: the twin with a runner-up comparator Q (rebuilt every step) and a keep gate cutting at 0.75 x the
runner-up so far, as ARC's does. Q takes memory cell 7's neuron; position 0 is read from the final runner-up's value.
Rules: algzoo.handbuilt_v27b (docstring). Layout as v26 (store 2, detector 4, keep 5, linear route 9, flip 1,
clocks 7/6/8, delay cells 0, 3, 4 -> ARC 0, 11, 10, delay cell 6 -> 3; delay cells 1, 2, 5 and Q -> 12-15 by activity
correlation).
Constants read off ARC: 4 (memory imprint 1, keep share 0.25, memory fade 0.99, keep cut 0.75 of the runner-up)
+ 16 loudness scales. Design choices fixed before scoring: eps 0.1 (r1a), delta 0.01, gq = g = 1e4."""
from algzoo.handbuilt_v27b import build_v27b
from algzoo.handbuilt_v26 import to_arc


def weights():
    return to_arc(build_v27b(), fixed={3: "mem6"}, free=(1, 2, 5, 7))
