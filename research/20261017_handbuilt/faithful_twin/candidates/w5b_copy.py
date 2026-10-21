"""w5b, variant 3 (24 neurons): w5b_slot7 plus a copy of the clocks (copy_share = 0.5): three helpers U0-U2 whose
rows equal the clocks' rows, so U equals the clocks at every step; each clock takes half of its rotation from U instead
of from the clocks. Same function exactly. Extra neurons U0-U2: job 8 as scored (a nudge of the hiding slots leaves
half of it in the clocks a step later); not ARC's way (a split of one rotation, not a rebuild from store, gap and late
gate). From variant 1: w4b_margin (rules in algzoo.handbuilt_w4b) with the hiding wave put back in ARC's
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

_W = build_w5b(copy_share=0.5)
MAP, LAYOUT = w5b_map(*_W)


def weights():
    return _W[:3]
