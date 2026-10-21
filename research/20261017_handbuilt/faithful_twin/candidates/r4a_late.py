"""r4a, variant 1: the twin's rules with a 2-D growing hiding wave (three clock neurons, no flip neuron), r3b's
runner-up comparator in slot 8 and keep cut against it in slot 5, and the freed neuron as a late gate in slot 9
carrying the switch-off push into memory cell 0. Rules: algzoo.handbuilt_v4a (module docstring).
Constants read off ARC: 4 (memory imprint 1, keep share 0.25, memory fade 0.99, keep cut 0.75 of the runner-up)
+ 16 loudness scales. Wave constants (theta 20 deg, phi 70 deg, growth 4) and gains are design rules, not read off ARC."""
from algzoo.handbuilt_v4a import build_v4a, to_arc


def weights():
    return to_arc(build_v4a())
