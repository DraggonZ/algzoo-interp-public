"""w4b, variant 1 (17 neurons): r3b's network (runner-up comparator Q, keep gate cutting at 0.75 of the runner-up) plus
a late gate LG in ARC's slot 9 (on one step after every record, holding the jump; it carries the switch-off push into
memory cell 0 that the linear route carried before). The linear route, which held slot 9, becomes a helper in no slot.
Layout otherwise as r3b_qx (flip 1, clocks 7/6/8, delay cells 0, 3, 4, 6 -> ARC 0, 11, 10, 3; delay cells 1, 2, 5 and Q
-> 12-15 by activity correlation). Rules: algzoo.handbuilt_w4b.
Extra neuron and its job: LG for job 5 (late gate), the 16-neuron build had no free neuron for it.
Constants read off ARC: 4 (memory imprint 1, keep share 0.25, memory fade 0.99, keep cut 0.75 of the runner-up)
+ 16 loudness scales (MAP). Design choices fixed before scoring: eps 0.1, delta 0.01, gq = g = 1e4."""
from algzoo.handbuilt_w4b import build_w4b, arc_map

_W = build_w4b(late=True, level=False, refs=False)
MAP, LAYOUT = arc_map(*_W)


def weights():
    return _W[:3]
