"""r2b, variant 3: r2b-2 (r2b_mix) with the linear route (ARC slot 9) holding L + nu x delay cell 7, its readers
correcting for it. The memory block then contains a loop from the last delay cell to the first (weight nu), which the
path through the linear route cancels on clean inputs: the eight memory neurons alone turn by 45, 90 and 135 degrees
per step with size (nu x 0.99^7)^(1/8). nu = 0.82^8 / 0.99^7 = 0.219, so that size is 0.82, the size per step of ARC's
slowest turning memory mode (r1a notes, `r1a/a2_loop.log`). Rules: algzoo.handbuilt_v26 (loop_through_linear).
Constants read off ARC: 5 (memory imprint 1, keep share 0.25, memory fade 0.99, keep cut 0.75 of the runner-up,
slowest memory turning size 0.82) + 16 loudness scales. Design choices fixed before scoring: eps 0.1, mixing weight 1."""
from algzoo.handbuilt_v26 import build_v26, mix_memory, loop_through_linear, to_arc

NU = 0.82 ** 8 / 0.99 ** 7


def weights():
    return to_arc(loop_through_linear(mix_memory(build_v26(), into="mem6", add="mem3", mu=1.0), NU))
