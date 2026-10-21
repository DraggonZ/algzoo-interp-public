# Noise control: is 'dense' just 'imprecise weights'? (2026-10-18)

Status: done (`run.py`, `run.log`). A concern, tested: random relative noise on the hand-built networks' weights, then the same density measures as for ARC.

| network | accuracy | near ties lost to the third | lead maps covering half | tree on the top three: accuracy (always-guess) |
|---|---|---|---|---|
| ARC (trained) | 95.4% | 20.8% | 7,955 | 85.8% (79.2%) |
| robust hand-built (28 neurons), no noise | 95.9% | 7.3% | 48 | 98.9% (92.7%) |
| same, noise 0.1% | 94.2-95.0% | 12-14% | 54-56 | 97.7-97.9% (86-88%) |
| same, noise 0.3% | 90.6-93.1% | 18-23% | 75-80 | 96.7-96.8% (77-82%) |
| same, noise 1% | 78.9-85.8% | 30-35% | 238-241 | 94.4-95.6% (66-71%) |
| same, noise 3% | 57.8-69.7% | 32-33% | 372-410 | 93.8-95.3% (67-68%) |
| fragile hand-built (16 neurons), noise 0.1-3% | 9-12% (broken) | — | 308-583 | — |

At the trained network's accuracy (noise 0.1%), the noisy robust hand-built network keeps about 55 maps and a tree that explains all but 2% of its near-tie decisions; ARC needs about 8,000 maps and leaves 14% unexplained. Even with noise that drops it to 79-86% accuracy, it stays far simpler than ARC (about 240 maps, 4-6% unexplained). The fragile hand-built network breaks under any noise and is no control. Prediction (written first): the maps count rises with noise, and the tree stays more predictive than for ARC; right, though the rise is much smaller than expected.

Reading: imprecise weights alone do not make the clean solution look like the trained one. ARC's dense close calls are not just a clean algorithm with noisy weights. Caveat: one robust hand-built network, random noise rather than structured errors.
