"""w7b, variant 1 (24 neurons): w6b_relay (rules in algzoo.handbuilt_w6b, w5b, w4b) with the hiding wave's sign flip
(180 degrees, one neuron) replaced by a second turning pair at 135 degrees (three clocks), so the memory turns in two
modes. Rules in algzoo.handbuilt_w7b (docstring):
  wave per unit of K d at offset k from the largest: f(k) = (cos 45k + cos 135k) / 2 (f(0) = 1, f(1) = f(2) = 0 as the
      twin's forward-safe rule; the free coefficient chosen so that no offset other than +-4, +-8 spills, which every
      45 + 135 wave fixes at -1 and +1);
  every reader of the wave follows the twin's rules for each part (built as w6b once per part, combined by linearity).
MAP: as w6b, slot 3 = the 45-degree clocks, slot 12 = the 135-degree clocks (the free memory slot whose ARC neuron
takes the largest share of ARC's 125-degree mode), the delay cell w6b had in slot 12 (mem2) becomes a helper.
Extra neurons over 16 (8): w6b's six (late gate, C7, C1, C6, CR, H7) and two for the second turning pair (job 9).
Constants read off ARC: 6 (memory imprint 1, keep share 0.25, memory fade 0.99, keep cut 0.75 of the runner-up, ARC's
neuron 6's and 1's weights on the previous largest two steps after a record, 0.23 and 0.95) + 16 loudness scales (MAP)
+ the slot choice for the 135-degree pair (an analysis read of ARC's modes).
Design choices fixed before scoring: as w6b, plus the second pair's angle 135 degrees (the angle that, with 45, keeps an
8-step wave with f(1) = f(2) = 0; ARC's nearest is 125-127) and a = p = 1/2, b = q = 0."""
from algzoo.handbuilt_w7b import build_w7b, w7b_map

_W = build_w7b()
MAP, LAYOUT = w7b_map(*_W)


def weights():
    return _W[:3]
