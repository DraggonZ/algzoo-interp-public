"""w4b, variant 4 (21 neurons): w4b_cancel with the hiding margin delivered through C1. The detector starts only
1 x the jump of the hiding wave one step after a record; the margin (kappa = 0.5 of the jump) enters the wave two steps
after, read as (C7 - C1) / (g c1): C1 holds the late gate's push, C7 does not. If the number two steps after the
record clips C1 (it is the new second-largest), the margin shrinks with it. Readout rows read the margin directly for
a record at step 7 or 8. So slot 1 carries the late gate's push into the hiding, as ARC's neuron 1 does.
From w4b_cancel: w4b_hide plus a helper CR that cancels the late gate's push on C1 and C6 when the number
after a record is itself a record (the push then belongs to a largest already overtaken). As in w4b_hide: three
comparators in ARC's hiding slots, so that each hiding slot holds a comparator rebuilt every step with ARC's
reference, and the four hiding-wave neurons (flip, three clocks) are helpers in no slot.
  slot 7: C7 = g relu(largest so far - x), read by every neuron that read the clocks' shared comparator level
          (Q, keep gate, linear route, readout of positions 0 and 9);
  slot 8: Q, the runner-up comparator (r3b);
  slot 6: C6 = g relu(store - 0.23 LG - x): previous largest one step after a record, largest minus 0.23 of the jump
          two steps after; read by nothing;
  slot 1: C1 = g relu(largest so far - 0.95 LG - x): largest one step after, about the previous largest two steps after;
          its difference from C7 carries the hiding margin into the wave (above).
Memory slots: delay cells 0, 3, 4, 6 -> ARC 0, 11, 10, 3; delay cells 1, 2, 5 and the linear route -> 12-15 by activity
correlation. Rules: algzoo.handbuilt_w4b.
Extra neurons and their jobs: LG (job 5, late gate; also the push of job 6), C7 (job 7: the comparator with the largest
that the wave-carrying clocks could not be on their own), C6 and C1 (jobs 6 and 7: ARC's references and the late gate's
push two steps after a record), CR (job 7: without it a record just before the record moves slot 1's reference).
Design choice fixed before scoring: CR's gain k = 1000 (the detector's gain a).
Constants read off ARC: 6 (memory imprint 1, keep share 0.25, memory fade 0.99, keep cut 0.75 of the runner-up,
ARC's neuron 6's and neuron 1's weights on the previous largest two steps after a record, 0.23 and 0.95)
+ 16 loudness scales (MAP). Design choices fixed before scoring: eps 0.1, delta 0.01, gq = g = 1e4."""
from algzoo.handbuilt_w4b import build_w4b, arc_map

_W = build_w4b(late=True, level=True, refs=True, cancel=True, margin=True)
MAP, LAYOUT = arc_map(*_W)


def weights():
    return _W[:3]
