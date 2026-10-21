# Training from a hand-built network (2026-10-18; approved exception to the no-optimization rule)

Status: done (`run.py`, `run.log`; partial float64 run in `run_fp64_partial.log`). Copies of the hand-built networks were trained further; the hand-built networks themselves are unchanged. Exact rescaling first (same function), then AdamW, batch 65,536, 3,000 steps.

| start | rate | steps | accuracy | near ties lost to the third | lead maps covering half | tree's share of the possible gain |
|---|---|---|---|---|---|---|
| v19 (16 neurons) | — | 0 | 94.2% | 7.2% | 46 | 91% |
| | 1e-5 | 100 / 1,000 / 3,000 | 90.9 / 92.9 / 91.1% | 36 / 34 / 35% | 374 / 529 / 1,055 | 61 / 62 / 59% |
| | 1e-4 | 100 / 1,000 / 3,000 | 46.8 / 58.3 / 66.6% | 29 / 35 / 38% | 4,355 / 4,248 / 5,293 | 58 / 61 / 61% |
| v21 (28 neurons) | — | 0 | 95.9% | 7.3% | 48 | 84% |
| | 1e-5 | 100 / 1,000 / 3,000 | 95.0 / 96.9 / 97.9% | 26 / 16 / 11% | 86 / 217 / 301 | 69 / 43 / 33% |
| | 1e-4 | 100 / 1,000 / 3,000 | 96.7 / 98.1 / 98.3% | 18 / 10 / 8% | 187 / 254 / 218 | 46 / 34 / 20% |

Prediction (written first: accuracy rises, maps rise toward thousands, tree's gain falls): right for the tree's gain, partly right for the maps, wrong for v19's accuracy.

## Reading

- The fragile 16-neuron hand-built network (margins tiny against its pushes) is broken by any gradient step and recovers as a dense network.
- The robust 28-neuron one improves quickly, from 95.9% to 98.3%, better than any trained 16-neuron network and near the trained 32-neuron ones. On the way its close calls stop being predictable from the top three (the tree's share of the gain falls from 84% to 20%, the trained networks' range), while its number of distinct lead maps grows only to about 200-300 (trained: 1,400-36,000).
- So the clean structure is not where training wants to stay: the first steps trade simple close calls for accuracy, making them depend on the smaller numbers. Training prefers dense close calls because they pay: context-dependent corrections raise accuracy, and nothing in the loss rewards keeping close calls simple. Caveat: one robust starting network, 3,000 steps.
