"""w4b, variant 2 (20 neurons): w4b_late plus three comparators in ARC's hiding slots, so that each hiding slot holds a
comparator rebuilt every step with ARC's reference, and the four hiding-wave neurons (flip, three clocks) become
helpers in no slot.
  slot 7: C7 = g relu(largest so far - x), read by every neuron that read the clocks' shared comparator level
          (Q, keep gate, linear route, readout of positions 0 and 9);
  slot 8: Q, the runner-up comparator (r3b);
  slot 6: C6 = g relu(store - 0.23 LG - x): previous largest one step after a record, largest minus 0.23 of the jump
          two steps after; read by nothing;
  slot 1: C1 = g relu(largest so far - 0.95 LG - x): largest one step after, about the previous largest two steps after;
          read by nothing.
Memory slots: delay cells 0, 3, 4, 6 -> ARC 0, 11, 10, 3; delay cells 1, 2, 5 and the linear route -> 12-15 by activity
correlation. Rules: algzoo.handbuilt_w4b.
Extra neurons and their jobs: LG (job 5, late gate; also the push of job 6), C7 (job 7: the comparator with the largest
that the wave-carrying clocks could not be on their own), C6 and C1 (jobs 6 and 7: ARC's references and the late gate's
push two steps after a record).
Constants read off ARC: 6 (memory imprint 1, keep share 0.25, memory fade 0.99, keep cut 0.75 of the runner-up,
ARC's neuron 6's and neuron 1's weights on the previous largest two steps after a record, 0.23 and 0.95)
+ 16 loudness scales (MAP). Design choices fixed before scoring: eps 0.1, delta 0.01, gq = g = 1e4."""
from algzoo.handbuilt_w4b import build_w4b, arc_map

_W = build_w4b(late=True, level=True, refs=True)
MAP, LAYOUT = arc_map(*_W)


def weights():
    return _W[:3]
