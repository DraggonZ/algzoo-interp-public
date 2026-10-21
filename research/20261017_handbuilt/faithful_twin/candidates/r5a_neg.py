"""r5a, variant 3: r4a_ring (2-D growing hiding wave, rho 4; runner-up comparator; keep cut against it; late gate
writing the backward wave; memory ring with readout compensation) with the ring's loop weight negative: -0.062 instead
of +0.062. Same loop size (ring modes of modulus 0.70, turning by 45, 90, 135 degrees), but what the ring carries
(position 0's cell at the last step, backward writes from cells that held no position yet) now raises the slot it lands
in instead of lowering it; a lowered true second hands the answer to position 0, which scores 0.99 x the runner-up.
Rules: algzoo.handbuilt_v4a. Layout and scale: algzoo.handbuilt_v5a (RMS, as r4a).
Constants read off ARC: 4 (memory imprint 1, keep share 0.25, memory fade 0.99, keep cut 0.75 of the runner-up)
+ 16 loudness scales."""
from algzoo.handbuilt_v5a import build_v4a, to_arc


def weights():
    return to_arc(build_v4a(bw_late=True, ring_w=-0.062, ring_comp=True))
