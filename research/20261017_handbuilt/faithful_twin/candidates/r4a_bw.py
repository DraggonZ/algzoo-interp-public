"""r4a, variant 2: r4a_late (2-D growing hiding wave, runner-up comparator in slot 8, keep cut against it, late gate in
slot 9) with the late gate carrying the record's backward hiding wave into the memory: one step after a record it writes
g(-(j+1)) x the jump into delay cell j+1 (the detector wrote the same values one step earlier, one cell back), plus the
switch-off push into cell 0. Silencing the late gate then costs 19 points (r4a-c). Rules: algzoo.handbuilt_v4a.
Constants read off ARC: 4 (memory imprint 1, keep share 0.25, memory fade 0.99, keep cut 0.75 of the runner-up)
+ 16 loudness scales."""
from algzoo.handbuilt_v4a import build_v4a, to_arc


def weights():
    return to_arc(build_v4a(bw_late=True))
