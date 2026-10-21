"""w5b, variant 2 (21 neurons): w5b_slot7 with readout_level=True: the readout takes the level it needs (last
number at position 9, runner-up read at position 0, margin of a record at step 7) from the three clocks in slot 7, as the
twin's readout did, instead of from the helper C7; recurrent readers stay on C7. From variant 1: w4b_margin (rules in algzoo.handbuilt_w4b) with the hiding wave put back in ARC's
hiding slots. Rules in algzoo.handbuilt_w5b:
  level-free reads: every reader outside the wave (linear route, readout rows) takes only the wave from the four
      hiding-wave neurons (clock0-2, flip): its weights on them change by lambda (1, 1, 1, 0), lambda = -(their sum)/3,
      and C7 (comparator with the largest) takes the level lost. Silencing the wave neurons then removes the wave only.
  map: ARC slot 7 (compares with the largest at both delays) = the four wave neurons together, equal weights (their
      level is a comparator with the largest); C7 is a helper in no slot. Slots 1, 6, 8: C1, C6, Q as in w4b.
Extra neurons over 16 (5): late gate (job 5), C7 (holds the arriving number for memory, keep gate and runner-up
comparator, so the hiding slots can be silenced without erasing it; hiding test of jobs 7-8), C1 and C6 (jobs 6 and 7:
ARC's references and the late gate's push), CR (job 7: cancels the push when the next number is a record).
Constants read off ARC: 6 (memory imprint 1, keep share 0.25, memory fade 0.99, keep cut 0.75 of the runner-up,
ARC's neuron 6's and 1's weights on the previous largest two steps after a record, 0.23 and 0.95) + 16 loudness scales
(MAP). Design choices fixed before scoring: eps 0.1, delta 0.01, gq = g = 1e4, CR gain 1000, margin 0.5 of the jump."""
from algzoo.handbuilt_w5b import build_w5b, w5b_map

_W = build_w5b(readout_level=True)
MAP, LAYOUT = w5b_map(*_W)


def weights():
    return _W[:3]
