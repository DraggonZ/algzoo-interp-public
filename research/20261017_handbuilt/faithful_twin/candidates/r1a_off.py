"""r1a, variant 1: the twin with the largest's memory cell switched off (eps = 0.1), relabelled so memory cells 0, 3, 4
sit in ARC's slots 0, 11, 10 (algzoo.handbuilt_v25, docstring there).

Rules: v23's (twin settings: imprint 1, keep share 0.25, fade 0.99, read off ARC) plus one: one step after a record the
detector adds eps = 0.1 of the jump to the linear route (a tenth, chosen before scoring: it keeps 80% of the hiding margin
kappa = 0.5 for a record at step 8), so the largest's own memory cell is pushed below zero.
Constants read off ARC: 3 (imprint, keep share, fade) + 16 loudness scales (activity RMS; no effect on decisions)."""
from algzoo.handbuilt_v25 import build_v25, to_arc


def weights():
    return to_arc(build_v25(eps=0.1, w=0.0))
