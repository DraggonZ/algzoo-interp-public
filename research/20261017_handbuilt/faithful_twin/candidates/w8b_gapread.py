"""w8b, variant 1 (24 neurons): w7b_twopair (rules in algzoo.handbuilt_w7b, w6b, w5b, w4b) with every clock (slot 3's
45-degree clocks, slot 12's 135-degree clocks) reading the wave's start and margin from three hiding comparators (H7,
C1, C6) and no longer from C7. Rule in algzoo.handbuilt_w8b (docstring): the new weights keep each clock's level
weight, its part one step after a record and its part two steps after, so the clocks' input is unchanged while no
number clips; one step after a record the start then arrives as C1 - C6 = g x (largest - second-largest).
MAP: w7b's. Extra neurons over 16 (8): as w7b.
Constants read off ARC: 6, as w7b (memory imprint 1, keep share 0.25, memory fade 0.99, keep cut 0.75 of the runner-up,
ARC's neuron 6's and 1's weights on the previous largest two steps after a record, 0.23 and 0.95) + 16 loudness scales
(MAP) + w7b's slot choice. No new constant.
"""
from algzoo.handbuilt_w8b import build_w8b, w8b_map

_W = build_w8b()
MAP, LAYOUT = w8b_map(*_W[:4])


def weights():
    return _W[:3]
