"""Neuron 6's weights from the multi-step rule against ARC's, weight by weight; and a check of where the rule's input
for neuron 6 differs from ARC's: by whether the late gate was on at the previous step."""
import numpy as np
import torch as th
from harness import *  # noqa
import step8_lategate as s8
import step11_more as s11
import step12_rollout as s12  # noqa: runs the multi-step fit (prints)

Cn = np.load("rule_rollout_constants.npy")
co = {n: Cn[i] for i, n in enumerate(HID)}
fitted = s8.weights_from_rule(s11.feat, co)
WH = simlib.WH
print("\nneuron 6, weight by weight (source: ARC / from rule):")
print("  " + ", ".join(f"{k}: {WH[6, k]:+.2f}/{fitted[6][k]:+.2f}" for k in range(16)))
with th.no_grad():
    pre = s12.rollout(s12.Ffit, s12.C).cpu().numpy()
g9 = np.concatenate([np.zeros((len(Xf), 1), bool), Hf[:, :-1, 9] > 0], 1)
on = Pf[:, :, 6] > 0
for lab, m in [("late gate on at the previous step", g9 & on), ("late gate off", ~g9 & on)]:
    e = pre[:, :, 1][m] - Pf[:, :, 6][m]
    print(f"  neuron 6 on, {lab}: {m.mean():.3f} of steps; rule error mean {e.mean():+.3f}, rms {np.sqrt((e ** 2).mean()):.3f}")
    gv = Hf[:, :-1, 9][(g9 & on)[:, 1:]] if lab.startswith("late gate on") else None
