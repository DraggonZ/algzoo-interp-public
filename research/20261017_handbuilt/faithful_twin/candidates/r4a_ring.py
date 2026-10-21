"""r4a, variant 3 (weak form, scored for the record): r4a_bw plus a memory ring (last delay cell back into the linear
route, w = 0.062, the smallest round value whose ring modes reach modulus 0.70) and the readout adding back the current
largest's backward wave that the ring carries. Rules: algzoo.handbuilt_v4a (ring_w, ring_comp).
Constants read off ARC: 4 (memory imprint 1, keep share 0.25, memory fade 0.99, keep cut 0.75 of the runner-up)
+ 16 loudness scales; the band's modulus 0.7 sets w."""
from algzoo.handbuilt_v4a import build_v4a, to_arc


def weights():
    return to_arc(build_v4a(bw_late=True, ring_w=0.062, ring_comp=True))
