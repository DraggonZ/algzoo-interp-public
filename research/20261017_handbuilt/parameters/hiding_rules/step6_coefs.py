"""Rule L1's constants, compared with ARC's weights where a term maps onto one source: the store and detector terms
(the rule's store and detector values) against ARC's weights from neurons 2 and 4; the feedback terms against ARC's
weights from the hiding neurons; x against the input weight."""
import io, contextlib
import numpy as np
from harness import *  # noqa
import step3_recursive as s3
import step5_lean as s5

with contextlib.redirect_stdout(io.StringIO()):
    co, rec = s3.fit_group(s5.make(["sd", "xprev", "lategate"]), "L1")
WI, WH = simlib.WI, simlib.WH
labels = ["x", "m1", "m2", "m1 wave cos", "m1 wave sin", "store (rule)", "detector (rule)", "previous x", "late gate (prev jump)",
          "fb from 1", "fb from 6", "fb from 7", "fb from 8"]
for n in HID:
    c = co[n]
    print(f"\nneuron {n}:")
    print("  " + ", ".join(f"{l} {v:+.3f}" for l, v in zip(labels, c)))
    print(f"  compare: input weight {WI[n]:+.3f}; ARC from store {WH[n, 2]:+.3f}, from detector {WH[n, 4]:+.3f}; "
          f"from 1, 6, 7, 8: {WH[n, 1]:+.3f} {WH[n, 6]:+.3f} {WH[n, 7]:+.3f} {WH[n, 8]:+.3f}")
