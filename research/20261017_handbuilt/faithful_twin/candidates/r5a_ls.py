"""r5a, variant 1: r4a_ring (2-D growing hiding wave, runner-up comparator, keep cut against it, late gate writing the
backward wave, memory ring w 0.062 with readout compensation) with each neuron put on ARC's scale by least squares
(k = <ours, ARC> / <ours, ours> over 20,000 lists) instead of by RMS. Same network, same answers.
Rules: algzoo.handbuilt_v4a (network), algzoo.handbuilt_v5a (layout and scale).
Constants read off ARC: 4 (memory imprint 1, keep share 0.25, memory fade 0.99, keep cut 0.75 of the runner-up)
+ 16 loudness scales (least squares)."""
from algzoo.handbuilt_v5a import build_v4a, to_arc


def weights():
    return to_arc(build_v4a(bw_late=True, ring_w=0.062, ring_comp=True), scale="ls")
