"""r1a, variant 2: r1a_off (the largest's memory cell switched off, eps = 0.1) plus a keep gate with a cut:
G = beta relu(x - theta m1), the linear route adding beta theta m1 (votes alpha x + beta max(x, theta m1)), theta = 0.44
= ARC's cut 0.75 of the runner-up x the median runner-up / largest so far (0.58). algzoo.handbuilt_v25, docstring there.
Constants read off ARC: 4 (imprint 1, keep share 0.25, fade 0.99, keep cut 0.75 of the runner-up) + 16 loudness scales."""
from algzoo.handbuilt_v25 import build_v25, to_arc


def weights():
    return to_arc(build_v25(eps=0.1, w=0.0, theta=0.75 * 0.58))
