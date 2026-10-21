"""r6a, variant 1: r5a_neg's network (2-D growing hiding wave, runner-up comparator, keep cut against it, late gate
writing the backward wave) with three changes (rules in algzoo.handbuilt_v6a):
- the memory ring is tapped one cell earlier: mem5 -> linear route, a 7-neuron ring, loop weight -0.087
  (r4a's rule: smallest weight in steps of 0.001 whose modes reach modulus 0.7; negative as in r5a), readout compensation for a 7-step ring;
- the readout of position 8 takes off the ring's last-step leak (position 1's cell) read from mem6;
- mem6 (ARC's slot 3, position 1's cell) adds the gap read off the hiding slots (level/gc - Q/gq, plus the detector
  at a record), gap_eff 1 in score units; position 1's readout takes the same hiding read of the final state off.
Layout and scale: algzoo.handbuilt_v5a.to_arc (RMS, as r4a and r5a).
Constants read off ARC: 4 (memory imprint 1, keep share 0.25, memory fade 0.99, keep cut 0.75 of the runner-up)
+ 16 loudness scales."""
from algzoo.handbuilt_v6a import build_v6a, to_arc


def weights():
    return to_arc(build_v6a(ring_tap=5, gap_eff=1.0, last_comp=True))
