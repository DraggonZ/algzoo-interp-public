"""r2b, variant 2: r1a-3 (switch-off eps = 0.1, keep cut theta = 0.44; the same network) with ARC's slot 3 holding
delay cell 6 plus delay cell 3 (one memory neuron in a mixed basis; its readers subtract the cell-3 part), so slot 3
carries the gap that delay cell 3 holds at ages 1-2. Delay cells 1, 2, 5, 7 go to ARC's 12-15 by activity correlation.
Rules: algzoo.handbuilt_v26 (mix_memory; build_v26 at delta 0, kappa 0.5 is handbuilt_v25's network).
Constants read off ARC: 4 (memory imprint 1, keep share 0.25, memory fade 0.99, keep cut 0.75 of the runner-up)
+ 16 loudness scales read off ARC's activity. Design choices fixed before scoring: eps 0.1, mixing weight 1."""
from algzoo.handbuilt_v26 import build_v26, mix_memory, to_arc


def weights():
    return to_arc(mix_memory(build_v26(), into="mem6", add="mem3", mu=1.0))
