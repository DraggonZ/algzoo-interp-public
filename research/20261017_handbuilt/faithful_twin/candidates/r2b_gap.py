"""r2b, variant 1: r1a-3 (switch-off eps = 0.1, keep cut theta = 0.44) with the hiding wave's forward spill set to
f(1) = f(2) = -0.05 instead of 0, so delay cell 6 carries the gap at ages 1-2; it goes to ARC's slot 3.
Rules: algzoo.handbuilt_v26 (docstring there), on top of handbuilt_v25 and handbuilt_v23.
Constants read off ARC: 4 (memory imprint 1, keep share 0.25, memory fade 0.99, keep cut 0.75 of the runner-up)
+ 16 loudness scales read off ARC's activity. Design choices fixed before scoring: eps 0.1, delta 0.05."""
from algzoo.handbuilt_v26 import build_v26, to_arc


def weights():
    return to_arc(build_v26(delta=0.05))
