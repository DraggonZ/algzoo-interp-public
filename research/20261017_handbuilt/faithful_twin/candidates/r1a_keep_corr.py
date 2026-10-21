"""r1a, variant 3: r1a_keep (switch-off eps = 0.1, keep cut theta = 0.44) with the five memory cells that have no
switching job placed in ARC's slots 3, 12, 13, 14, 15 by activity correlation (as match_twin.py placed all eight).
Same function as r1a_keep. algzoo.handbuilt_v25, docstring there.
Constants read off ARC: 4 (imprint 1, keep share 0.25, fade 0.99, keep cut 0.75 of the runner-up) + 16 loudness scales."""
from algzoo.handbuilt_v25 import build_v25, to_arc


def weights():
    return to_arc(build_v25(eps=0.1, w=0.0, theta=0.75 * 0.58), free_by_corr=True)
