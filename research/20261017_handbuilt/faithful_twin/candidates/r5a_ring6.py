"""r5a, variant 2: r4a_ring (2-D growing hiding wave, runner-up comparator, keep cut against it, late gate writing the
backward wave, memory ring w 0.062 with readout compensation) with the hiding wave growing 6-fold per step instead of
4-fold, which shrinks the backward writes g(-k) that the ring carries out of cells holding no position yet
(g(-1) 0.47 instead of 0.70). Clock gain and linear-route level follow v4a's scale rules (gc / rho^8 and
lf / |g(8)| kept as at rho 4). Rules: algzoo.handbuilt_v4a. Layout and scale: algzoo.handbuilt_v5a (RMS).
Constants read off ARC: 4 (memory imprint 1, keep share 0.25, memory fade 0.99, keep cut 0.75 of the runner-up)
+ 16 loudness scales."""
from algzoo.handbuilt_v5a import build_v4a, to_arc
from algzoo.handbuilt_v4a import wave_2d, g_of

RHO = 6.0
GC = 1e10 * (RHO / 4) ** 8
LF = 2e5 * abs(g_of(*wave_2d(rho_w=RHO), 8)) / abs(g_of(*wave_2d(rho_w=4.0), 8))


def weights():
    return to_arc(build_v4a(bw_late=True, ring_w=0.062, ring_comp=True, rho_w=RHO, gc=GC, lf=LF))
