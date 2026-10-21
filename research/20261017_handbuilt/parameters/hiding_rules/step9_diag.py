"""Two diagnostics: (a) when the late gate is on, are hiding neurons 1 and 6 on or off at the next step? (b) does rule
L1 shrink neuron 8's input relative to ARC's on its on-steps (slope of rule input on ARC input)?"""
import io, contextlib
import numpy as np
from harness import *  # noqa
import step3_recursive as s3
import step5_lean as s5

g9 = Hs[:, :-1, 9] > 0
for n in [1, 6]:
    pre_next = Ps[:, 1:, n]
    print(f"(a) steps after the late gate is on ({g9.mean():.3f} of steps): neuron {n} is on at the next step in "
          f"{(pre_next[g9] > 0).mean():.3f} of them; its input there: median {np.median(pre_next[g9]):+.2f}, 10th percentile {np.percentile(pre_next[g9], 10):+.2f}")
    push = Hs[:, :-1, 9][g9] * simlib.WH[n, 9]
    print(f"    the late gate's push on it there: median {np.median(push):+.2f}; without the push its input would be "
          f"positive in {((pre_next[g9] - push) > 0).mean():.3f} of them")
with contextlib.redirect_stdout(io.StringIO()):
    feat = s5.make(["sd", "xprev", "lategate"])
    co, _ = s3.fit_group(feat, "L1")
Ff = {n: feat(Xf, n) for n in HID}
rule_pre = np.zeros((len(Xf), 10, 4)); prev = np.zeros((len(Xf), 4))
for t in range(10):
    for i, n in enumerate(HID):
        rule_pre[:, t, i] = np.concatenate([Ff[n][:, t], prev], -1) @ co[n]
    prev = np.maximum(rule_pre[:, t], 0)
for i, n in enumerate(HID):
    on = Pf[:, :, n] > 0
    a, r = Pf[:, :, n][on], rule_pre[:, :, i][on]
    rec = a - simlib.WI[n] * Xf[on]; rr = r - simlib.WI[n] * Xf[on]
    print(f"(b) neuron {n}, on-steps: slope of the rule's recurrent part on ARC's {np.polyfit(rec, rr, 1)[0]:.3f}; "
          f"rms of ARC's recurrent part {np.sqrt((rec ** 2).mean()):.2f}, rule error there {np.sqrt(((rr - rec) ** 2).mean()):.2f}")
