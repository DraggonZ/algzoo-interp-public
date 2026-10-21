"""Weights from the richest activity rule (B + late gate + previous number, 29 constants per neuron) and from L1 + kept
wave (15), compared as in step7/step8."""
import numpy as np
from harness import *  # noqa
import step3_recursive as s3
import step4_ablate as s4
import step5_lean as s5
import step8_lategate as s8

WH = simlib.WH
for lab, feat in [("richest (29 per neuron)", s4.make(["sd", "kept", "modes", "lategate", "xprev"])),
                  ("L1 + kept wave (15 per neuron)", s5.make(["sd", "xprev", "lategate"], True))]:
    co, _ = s3.fit_group(feat, lab)
    fitted = s8.weights_from_rule(feat, co)
    for n in HID:
        a, f = WH[n], fitted[n]
        print(f"  weights for {n}: cosine {a @ f / np.linalg.norm(a) / np.linalg.norm(f):.3f}, relative distance {np.linalg.norm(f - a) / np.linalg.norm(a):.2f}")
    for rows in [[1], [6], [7], [8], HID]:
        P = [w.copy() for w in s8.W0]
        for n in rows:
            P[1][n] = fitted[n]
        l = s8.run_g(P)
        print(f"  rows {rows} from the rule: accuracy {100 * ((l.argmax(-1) == s8.Y).float().mean().item() - s8.acc0):+.2f} points, KL {(s8.pb * (s8.lb - l)).sum(-1).mean().item():.3f}")
