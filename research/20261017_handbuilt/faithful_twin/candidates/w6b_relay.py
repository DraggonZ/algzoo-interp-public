"""w6b, variant 1 (22 neurons): w5b_slot7 (rules in algzoo.handbuilt_w5b, handbuilt_w4b) with the hiding wave's start
relayed through a new comparator H7 in ARC's hiding slot 7, and the wave's rotation held in the four former wave neurons,
mapped to ARC's memory slot 3. Rules in algzoo.handbuilt_w6b (docstring):
  H7_{t+1} = relu(g (m1(t) - x_{t+1}) + c d_t), c = 1: a comparator with the largest that also carries a record's jump for
      one step; rebuilt every step from the store, the detector and the input.
  the wave neurons take the start from H7 - C7 one step later instead of from the detector; every reader of the wave
      takes the start's share from H7 - C7, so the function is w5b_slot7's.
Extra neurons over 16 (6): late gate (job 5), C7 (holds the arriving number for the memory, keep gate and runner-up
comparator), C1 and C6 (jobs 6 and 7), CR (job 7), H7 (jobs 7 and 8: the hiding slot that starts the wave and is
rebuilt every step).
Constants read off ARC: 6 (memory imprint 1, keep share 0.25, memory fade 0.99, keep cut 0.75 of the runner-up, ARC's
neuron 6's and 1's weights on the previous largest two steps after a record, 0.23 and 0.95) + 16 loudness scales (MAP).
Design choices fixed before scoring: as w5b_slot7, plus H7's start gain c = 1."""
from algzoo.handbuilt_w6b import build_w6b, w6b_map

_W = build_w6b()
MAP, LAYOUT = w6b_map(*_W)


def weights():
    return _W[:3]
